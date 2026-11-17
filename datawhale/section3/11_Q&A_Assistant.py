"""
Author: kaic
Date: 2026-11-16 14:00
LastEditors: kai2dot0 41042242+kai2dot0@users.noreply.github.com
LastEditTime: 2026-11-16 14:00
Copyright (c) 2026 by kaic, All Rights Reserved.
"""

"""
智能文档问答助手 - 基于HelloAgents的智能文档问答系统

这是一个完整的PDF学习助手应用，支持：
- 加载PDF文档并构建知识库
- 智能问答（基于RAG）
- 学习历程记录（基于Memory）
- 学习回顾和报告生成
"""

import os
import time
import json
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4

from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM
from hello_agents.memory import MemoryConfig
from hello_agents.memory.embedding import get_text_embedder
from hello_agents.memory.rag.pipeline import (
    _preprocess_markdown_for_embedding,
    load_and_chunk_texts,
)
from hello_agents.tools import MemoryTool, RAGTool
import gradio as gr
import numpy as np

APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR.parents[1] / ".env")


def use_query_points(store):
    """将旧框架的 search 调用适配到当前 Qdrant 的 query_points 接口。"""
    def search(*, query_vector, **kwargs):
        """保留框架的过滤条件和搜索参数，并返回原有的命中列表格式。"""
        return store.client.query_points(query=query_vector, **kwargs).points

    store.client.search = search


def tool_succeeded(result: str) -> bool:
    """依据当前框架的成功标记识别写入结果，避免将错误字符串视为成功。"""
    return isinstance(result, str) and result.startswith("✅")


class PDFLearningAssistant:
    """智能文档问答助手"""

    def __init__(self, user_id: str = "default_user"):
        """初始化学习助手

        Args:
            user_id: 用户ID，用于隔离不同用户的数据
        """
        self.user_id = user_id.strip() or "web_user"
        self.session_id = f"session_{uuid4().hex}"
        for name in ("MODELSCOPE_BASE_URL", "MODELSCOPE_API_KEY"):
            if not os.getenv(name, "").strip():
                raise ValueError(f"请在项目根目录 .env 中配置 {name}")
        self.llm = HelloAgentsLLM(
            provider="modelscope",
            model=os.getenv("MODELSCOPE_MODEL_ID", "Qwen/Qwen3.5-35B-A3B"),
            base_url=os.environ["MODELSCOPE_BASE_URL"],
            api_key=os.environ["MODELSCOPE_API_KEY"],
        )

        # 初始化工具
        self.memory_tool = MemoryTool(
            user_id=self.user_id,
            memory_config=MemoryConfig(storage_path=str(APP_DIR / "memory_data")),
        )
        self.memory_tool.current_session_id = self.session_id
        self.namespace = f"pdf_{self.user_id}"
        self.rag_tool = RAGTool(
            knowledge_base_path=str(APP_DIR / "knowledge_base"),
            # 框架的分块 ID 不含命名空间；独立集合避免相同 PDF 跨用户覆盖。
            collection_name=f"pdf_{sha256(self.user_id.encode()).hexdigest()[:24]}",
            rag_namespace=self.namespace,
        )
        if not self.rag_tool.initialized:
            raise RuntimeError(f"知识库初始化失败：{self.rag_tool.init_error}")
        self.rag_tool.llm = self.llm

        # 仅适配本助手使用的存储实例，不修改第三方包或关闭兼容性检查。
        for memory in self.memory_tool.memory_manager.memory_types.values():
            store = getattr(memory, "vector_store", None)
            if store is not None:
                use_query_points(store)
        pipeline = self.rag_tool._pipelines[self.namespace]
        use_query_points(pipeline["store"])
        # 绕过旧框架中批次过大、数组转换错误及零向量回退的索引实现。
        pipeline["add_documents"] = self._add_documents
        # 内置高级检索会重新创建默认 LLM；替换为复用 ModelScope 的查询扩展。
        pipeline["search_advanced"] = self._search_advanced

        # 学习统计
        self.stats = {
            "session_start": datetime.now(),
            "documents_loaded": 0,
            "questions_asked": 0,
            "concepts_learned": 0,
        }

        # 当前加载的文档
        self.current_document = None

    def _add_documents(
        self, file_paths: List[str], chunk_size: int = 1000, chunk_overlap: int = 200,
    ) -> int:
        """复用框架解析和存储，每批最多编码十条，全部校验通过后才入库。"""
        chunks = load_and_chunk_texts(
            paths=file_paths, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            namespace=self.namespace, source_label="rag",
        )
        if not chunks:
            return 0

        store = self.rag_tool._pipelines[self.namespace]["store"]
        embedder = get_text_embedder()
        texts = [_preprocess_markdown_for_embedding(chunk["content"]) for chunk in chunks]
        vectors = []
        batch_size = 10  # 当前 Embedding 服务明确限制每个请求最多十条文本。
        print(f"[RAG] Embedding start: total_texts={len(texts)} batch_size={batch_size}")
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            # np.asarray 同时支持二维数组、向量数组列表和普通嵌套列表。
            batch_vectors = np.asarray(embedder.encode(batch), dtype=float)
            if batch_vectors.ndim == 1 and len(batch) == 1:
                batch_vectors = batch_vectors.reshape(1, -1)
            expected_shape = (len(batch), store.vector_size)
            if batch_vectors.shape != expected_shape:
                raise ValueError(
                    f"第 {start // batch_size + 1} 批向量数量或维度错误："
                    f"期望 {expected_shape}，实际 {batch_vectors.shape}，已停止入库"
                )
            if not np.isfinite(batch_vectors).all() or np.any(np.all(batch_vectors == 0, axis=1)):
                raise ValueError("Embedding 返回非有限值或零向量，已停止入库")
            vectors.extend(batch_vectors.tolist())
            print(f"[RAG] Embedding progress: {start + len(batch)}/{len(texts)}")

        metadata = [{
            **chunk.get("metadata", {}),
            "memory_id": chunk["id"],
            "user_id": self.user_id,
            "memory_type": "rag_chunk",
            "content": chunk["content"],
            "data_source": "rag_pipeline",
            "rag_namespace": self.namespace,
            "is_rag_data": True,
        } for chunk in chunks]
        ids = [chunk["id"] for chunk in chunks]
        if not (len(vectors) == len(metadata) == len(ids) == len(chunks)):
            raise ValueError("向量、文本块和元数据数量不一致，已停止入库")
        print(f"[RAG] Qdrant upsert start: n={len(vectors)}")
        if not store.add_vectors(vectors=vectors, metadata=metadata, ids=ids):
            raise RuntimeError("向量写入 Qdrant 失败")
        print(f"[RAG] Qdrant upsert done: {len(vectors)} vectors indexed")
        return len(chunks)

    def _search_advanced(
        self, query: str, top_k: int = 5, enable_mqe: bool = False,
        enable_hyde: bool = False, score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """使用同一个 LLM 扩展查询，并合并去重多个基础检索的结果。"""
        queries = [query]
        if enable_mqe:
            expansion = self.llm.invoke([
                {"role": "system", "content": "为文档检索改写问题，输出两条不同表述，每行一条，不要编号或回答问题。"},
                {"role": "user", "content": query},
            ])
            queries.extend(line.strip() for line in expansion.splitlines() if line.strip())
        if enable_hyde:
            queries.append(self.llm.invoke([
                {"role": "system", "content": "为检索生成一段可能包含答案的短文，保留问题中的关键术语。这段假设文本仅用于检索，不向用户作为事实展示。"},
                {"role": "user", "content": query},
            ]))
        search = self.rag_tool._pipelines[self.namespace]["search"]
        hits = {}
        for expanded_query in queries[:4]:
            for hit in search(expanded_query, top_k=top_k, score_threshold=score_threshold):
                key = str(hit["id"])
                if key not in hits or hit["score"] > hits[key]["score"]:
                    hits[key] = hit
        return sorted(hits.values(), key=lambda hit: hit["score"], reverse=True)[:top_k]

    def load_document(self, pdf_path: str) -> Dict[str, Any]:
        """加载PDF文档到知识库

        Args:
            pdf_path: PDF文件路径

        Returns:
            Dict: 包含success和message的结果
        """
        if not pdf_path or not Path(pdf_path).is_file():
            return {"success": False, "message": f"文件不存在: {pdf_path}"}
        if Path(pdf_path).suffix.lower() != ".pdf":
            return {"success": False, "message": "请选择 PDF 文件"}

        start_time = time.time()

        try:
            # 使用RAG工具处理PDF
            result = self.rag_tool.run(
                {
                    "action": "add_document",
                    "namespace": self.namespace,
                    "file_path": pdf_path,
                    "chunk_size": 1000,
                    "chunk_overlap": 200,
                }
            )
            if not tool_succeeded(result):
                return {"success": False, "message": result}

            process_time = time.time() - start_time

            # RAG工具返回的是字符串消息
            self.current_document = os.path.basename(pdf_path)
            self.stats["documents_loaded"] += 1

            # 记录到学习记忆
            memory_result = self.memory_tool.run(
                {
                    "action": "add",
                    "content": f"加载了文档《{self.current_document}》",
                    "memory_type": "episodic",
                    "importance": 0.9,
                    "event_type": "document_loaded",
                    "session_id": self.session_id,
                }
            )

            message = f"加载成功！(耗时: {process_time:.1f}秒)"
            if not tool_succeeded(memory_result):
                message += f"\n⚠️ 文档已入库，但学习记录保存失败：{memory_result}"
            return {
                "success": True,
                "message": message,
                "document": self.current_document,
            }
        except Exception as e:
            return {"success": False, "message": f"加载失败: {str(e)}"}

    def ask(self, question: str, use_advanced_search: bool = False) -> str:
        """向文档提问

        Args:
            question: 用户问题
            use_advanced_search: 是否使用 LLM 查询扩展后合并检索结果

        Returns:
            str: 答案
        """
        question = question.strip()
        if not question:
            return "⚠️ 问题不能为空"
        if not self.current_document:
            return "⚠️ 请先在‘开始使用’中加载 PDF 文档。"

        # 记录问题到工作记忆
        working_result = self.memory_tool.run(
            {
                "action": "add",
                "content": f"提问: {question}",
                "memory_type": "working",
                "importance": 0.6,
                "session_id": self.session_id,
            }
        )

        # 使用RAG检索答案
        answer = self.rag_tool.run(
            {
                "action": "ask",
                "namespace": self.namespace,
                "question": question,
                "limit": 5,
                "enable_advanced_search": use_advanced_search,
            }
        )
        if answer.startswith(("❌", "⚠️", "🤔")):
            return answer

        # 保存完整问答及文档名，便于之后回顾实际学过的内容。
        memory_result = self.memory_tool.run(
            {
                "action": "add",
                "content": f"文档《{self.current_document}》\n问题：{question}\n回答：{answer}",
                "memory_type": "episodic",
                "importance": 0.7,
                "event_type": "qa_interaction",
                "session_id": self.session_id,
            }
        )

        self.stats["questions_asked"] += 1
        for result in (working_result, memory_result):
            if not tool_succeeded(result):
                answer += f"\n\n⚠️ 学习记忆保存失败：{result}"

        return answer

    def add_note(self, content: str, concept: Optional[str] = None) -> str:
        """添加学习笔记

        Args:
            content: 笔记内容
            concept: 相关概念（可选）
        """
        content = content.strip()
        if not content:
            return "❌ 笔记内容不能为空"
        result = self.memory_tool.run(
            {
                "action": "add",
                "content": content,
                "memory_type": "semantic",
                "importance": 0.8,
                "concept": concept or "general",
                "session_id": self.session_id,
            }
        )

        if tool_succeeded(result):
            self.stats["concepts_learned"] += 1
        return result

    def recall(self, query: str, limit: int = 5) -> str:
        """回顾学习历程

        Args:
            query: 查询关键词
            limit: 返回结果数量

        Returns:
            str: 相关记忆
        """
        if not query.strip():
            return "⚠️ 请输入需要回顾的关键词"
        result = self.memory_tool.run(
            {"action": "search", "query": query, "limit": limit}
        )
        return result

    def get_stats(self) -> Dict[str, Any]:
        """获取学习统计

        Returns:
            Dict: 统计信息
        """
        duration = (datetime.now() - self.stats["session_start"]).total_seconds()

        return {
            "会话时长": f"{duration:.0f}秒",
            "加载文档": self.stats["documents_loaded"],
            "提问次数": self.stats["questions_asked"],
            "学习笔记": self.stats["concepts_learned"],
            "当前文档": self.current_document or "未加载",
        }

    def generate_report(self, save_to_file: bool = True) -> Dict[str, Any]:
        """生成学习报告

        Args:
            save_to_file: 是否保存到文件

        Returns:
            Dict: 学习报告
        """
        # 获取记忆摘要
        memory_summary = self.memory_tool.run({"action": "summary", "limit": 10})

        # 获取RAG统计
        rag_stats = self.rag_tool.run({"action": "stats", "namespace": self.namespace})

        # 生成报告
        duration = (datetime.now() - self.stats["session_start"]).total_seconds()
        report = {
            "session_info": {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "start_time": self.stats["session_start"].isoformat(),
                "duration_seconds": duration,
            },
            "learning_metrics": {
                "documents_loaded": self.stats["documents_loaded"],
                "questions_asked": self.stats["questions_asked"],
                "concepts_learned": self.stats["concepts_learned"],
            },
            "memory_summary": memory_summary,
            "rag_status": rag_stats,
        }

        # 保存到文件
        if save_to_file:
            report_file = APP_DIR / "learning_reports" / f"learning_report_{self.session_id}.json"
            try:
                report_file.parent.mkdir(parents=True, exist_ok=True)
                with open(report_file, "w", encoding="utf-8") as f:
                    json.dump(report, f, ensure_ascii=False, indent=2, default=str)
                report["report_file"] = str(report_file)
            except Exception as e:
                report["save_error"] = str(e)

        return report


def create_gradio_ui():
    """创建Gradio Web UI"""
    def init_assistant(user_id: str):
        """初始化当前浏览器会话的助手，并清空上一用户的界面历史。"""
        try:
            assistant = PDFLearningAssistant(user_id=user_id)
            return f"✅ 助手已初始化 (用户: {assistant.user_id})", assistant, [], ""
        except Exception as exc:
            return f"❌ 初始化失败：{exc}", None, [], ""

    def load_pdf(pdf_file, assistant) -> str:
        """加载PDF文件"""
        if assistant is None:
            return "❌ 请先初始化助手"

        if pdf_file is None:
            return "❌ 请上传PDF文件"

        # type="filepath" 返回路径字符串，不是需要读取 .name 的文件对象。
        result = assistant.load_document(str(pdf_file))

        if result["success"]:
            return f"✅ {result['message']}\n📄 文档: {result['document']}"
        else:
            return f"❌ {result['message']}"

    def chat(message: str, history: List, assistant, mode: str, advanced: bool) -> Tuple[str, List]:
        """按明确选择的模式回答问题，使用 Gradio 6 的消息字典格式。"""
        history = list(history or [])
        if not message.strip():
            return "", history
        try:
            if assistant is None:
                response = "❌ 请先初始化助手并加载文档"
            elif mode == "学习回顾":
                response = f"🧠 **学习回顾**\n\n{assistant.recall(message)}"
            else:
                response = f"💡 **回答**\n\n{assistant.ask(message, advanced)}"
        except Exception as exc:
            response = f"❌ 操作失败：{exc}"
        history.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ])
        return "", history

    def add_note_ui(note_content: str, concept: str, assistant) -> str:
        """添加笔记"""
        if assistant is None:
            return "❌ 请先初始化助手"

        if not note_content.strip():
            return "❌ 笔记内容不能为空"

        try:
            return assistant.add_note(note_content, concept or None)
        except Exception as exc:
            return f"❌ 笔记保存失败：{exc}"

    def get_stats_ui(assistant) -> str:
        """获取统计信息"""
        if assistant is None:
            return "❌ 请先初始化助手"

        stats = assistant.get_stats()
        result = "📊 **学习统计**\n\n"
        for key, value in stats.items():
            result += f"- **{key}**: {value}\n"
        return result

    def generate_report_ui(assistant):
        """生成报告并返回可下载的 JSON 文件。"""
        if assistant is None:
            return "❌ 请先初始化助手", None
        try:
            report = assistant.generate_report(save_to_file=True)
        except Exception as exc:
            return f"❌ 报告生成失败：{exc}", None
        if "save_error" in report:
            return f"❌ 报告保存失败：{report['save_error']}", None

        result = f"✅ 学习报告已生成\n\n"
        result += f"**会话信息**\n"
        result += f"- 会话时长: {report['session_info']['duration_seconds']:.0f}秒\n"
        result += f"- 加载文档: {report['learning_metrics']['documents_loaded']}\n"
        result += f"- 提问次数: {report['learning_metrics']['questions_asked']}\n"
        result += f"- 学习笔记: {report['learning_metrics']['concepts_learned']}\n"

        if "report_file" in report:
            result += f"\n💾 报告已保存至: {report['report_file']}"

        return result, report.get("report_file")

    # 创建Gradio界面
    with gr.Blocks(title="智能文档问答助手", analytics_enabled=False) as demo:
        # 每个浏览器会话独立持有助手，避免共享一个可变的全局实例。
        assistant_state = gr.State(None)
        gr.Markdown("""
        # 📚 智能文档问答助手

        基于HelloAgents的智能文档问答系统，支持：
        - 📄 加载PDF文档并构建知识库
        - 💬 智能问答（基于RAG）
        - 📝 学习笔记记录
        - 🧠 学习历程回顾
        - 📊 学习报告生成
        """)

        with gr.Tab("🏠 开始使用"):
            with gr.Row():
                user_id_input = gr.Textbox(
                    label="用户ID",
                    placeholder="输入你的用户ID（可选，默认为web_user）",
                    value="web_user",
                )
                init_btn = gr.Button("初始化助手", variant="primary")

            init_output = gr.Textbox(label="初始化状态", interactive=False)

            gr.Markdown("### 📄 加载PDF文档")
            pdf_upload = gr.File(
                label="上传PDF文件", file_types=[".pdf"], type="filepath"
            )
            load_btn = gr.Button("加载文档", variant="primary")
            load_output = gr.Textbox(label="加载状态", interactive=False)
            load_btn.click(load_pdf, inputs=[pdf_upload, assistant_state], outputs=[load_output])

        with gr.Tab("💬 智能问答"):
            gr.Markdown("### 向文档提问或回顾学习历程")
            mode = gr.Radio(["文档问答", "学习回顾"], value="文档问答", label="模式")
            advanced = gr.Checkbox(value=False, label="高级检索（查询扩展与假设答案，会增加模型调用）")
            chatbot = gr.Chatbot(label="对话历史", height=400)
            with gr.Row():
                msg_input = gr.Textbox(
                    label="输入问题",
                    placeholder="文档问答可直接提问；学习回顾请输入知识点关键词。",
                    scale=4,
                )
                send_btn = gr.Button("发送", variant="primary", scale=1)

            gr.Examples(
                examples=[
                    "什么是大语言模型？",
                    "Transformer架构有哪些核心组件？",
                    "如何训练大语言模型？",
                    "我之前学过什么内容？",
                    "回顾一下关于注意力机制的学习",
                ],
                inputs=msg_input,
            )

            msg_input.submit(
                chat, inputs=[msg_input, chatbot, assistant_state, mode, advanced], outputs=[msg_input, chatbot],
                concurrency_id="assistant",
            )
            send_btn.click(
                chat, inputs=[msg_input, chatbot, assistant_state, mode, advanced], outputs=[msg_input, chatbot],
                concurrency_id="assistant",
            )

        with gr.Tab("📝 学习笔记"):
            gr.Markdown("### 记录学习心得和重要概念")
            note_content = gr.Textbox(
                label="笔记内容", placeholder="输入你的学习笔记...", lines=3
            )
            concept_input = gr.Textbox(
                label="相关概念（可选）", placeholder="例如：transformer, attention"
            )
            note_btn = gr.Button("保存笔记", variant="primary")
            note_output = gr.Textbox(label="保存状态", interactive=False)
            note_btn.click(
                add_note_ui, inputs=[note_content, concept_input, assistant_state], outputs=[note_output]
            )

        with gr.Tab("📊 学习统计"):
            gr.Markdown("### 查看学习进度和统计信息")
            stats_btn = gr.Button("刷新统计", variant="primary")
            stats_output = gr.Markdown()
            stats_btn.click(get_stats_ui, inputs=[assistant_state], outputs=[stats_output])

            gr.Markdown("### 生成学习报告")
            report_btn = gr.Button("生成报告", variant="primary")
            report_output = gr.Textbox(label="报告状态", interactive=False)
            report_file = gr.File(label="下载学习报告", interactive=False)
            report_btn.click(generate_report_ui, inputs=[assistant_state], outputs=[report_output, report_file])

        init_btn.click(
            init_assistant, inputs=[user_id_input],
            outputs=[init_output, assistant_state, chatbot, load_output],
        )

    return demo


def main():
    """主函数 - 启动Gradio Web UI"""
    print("\n" + "=" * 60)
    print("智能文档问答助手")
    print("=" * 60)
    print("正在启动Web界面...\n")

    demo = create_gradio_ui()
    demo.queue().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        theme=gr.themes.Soft(), share=False, show_error=True,
    )


if __name__ == "__main__":
    main()
