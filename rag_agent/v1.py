from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
import json
import re
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os
from dotenv import load_dotenv

# LangChain & LangGraph
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# Retrieval
from rank_bm25 import BM25Okapi

# MLflow
import mlflow
import argparse
from typing import Final
import mlflow.langchain


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class LaptopChunk:
    """Represents a searchable chunk of laptop information."""
    chunk_id: str
    laptop_id: int
    content: str
    metadata: Dict[str, Any]
    token_count: int
    field_name: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "laptop_id": self.laptop_id,
            "content": self.content,
            "metadata": self.metadata,
            "token_count": self.token_count,
            "field_name": self.field_name
        }


@dataclass
class RetrievalResult:
    """Result from retrieval phase."""
    query: str
    chunks: List[LaptopChunk]
    scores: List[float]
    retrieval_method: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "chunks": [c.to_dict() for c in self.chunks],
            "scores": self.scores,
            "retrieval_method": self.retrieval_method
        }


@dataclass
class GenerationResult:
    """Result from generation phase."""
    answer: str
    citations: List[str]
    used_chunks: List[str]
    word_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "used_chunks": self.used_chunks,
            "word_count": self.word_count
        }


@dataclass
class CriticalAgentDecision:
    """Decision made by critical agent."""
    is_valid: bool
    unsupported_claims: List[str]
    supported_claims: List[str]
    reason: str
    action: str  # "accept", "reject", "regenerate"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "unsupported_claims": self.unsupported_claims,
            "supported_claims": self.supported_claims,
            "reason": self.reason,
            "action": self.action,
            "timestamp": self.timestamp
        }


@dataclass
class RAGResult:
    """Complete RAG pipeline result."""
    query: str
    retrieval: RetrievalResult
    generation: GenerationResult
    critical_decision: CriticalAgentDecision
    final_answer: str
    iteration_count: int
    total_latency: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "retrieval": self.retrieval.to_dict(),
            "generation": self.generation.to_dict(),
            "critical_decision": self.critical_decision.to_dict(),
            "final_answer": self.final_answer,
            "iteration_count": self.iteration_count,
            "total_latency": self.total_latency
        }


# ============================================================================
# AGENT STATE (LangGraph)
# ============================================================================

class AgentState(TypedDict):
    """State for LangGraph agent workflow."""
    query: str
    retrieved_chunks: List[Dict[str, Any]]  # Serialized chunks
    retrieval_scores: List[float]
    generated_answer: str
    citations: List[str]
    is_valid: bool
    unsupported_claims: List[str]
    supported_claims: List[str]
    decision_reason: str
    action: str
    iteration: int
    max_iterations: int
    final_answer: str
    decisions_log: List[Dict[str, Any]]


# ============================================================================
# INGESTION & CHUNKING
# ============================================================================

class ChunkingStrategy(ABC):
    """Abstract base for chunking strategies."""
    
    @abstractmethod
    def create_chunks(self, laptop_data: Dict[str, Any]) -> List[LaptopChunk]:
        """Create chunks from laptop data."""
        pass


class FieldBasedChunking(ChunkingStrategy):
    """
    Chunking strategy: one chunk per important field.
    
    Why: Maintains semantic coherence; easier citation tracking.
    How: Each technical field becomes a chunk with [laptop_id:field] format.
    """
    
    def __init__(self, min_tokens: int = 10, max_tokens: int = 120):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.important_fields = [
            "full_name", "producer", "model", "cpu", "cpu_mark", 
            "gpu", "ram", "disc", "display_size", "display_resolution",
            "display_hz", "display_tech", "weight", "price_in_dollar",
            "wifi_version", "bluetooth_version", "publishing_date"
        ]
    
    def _count_tokens(self, text: str) -> int:
        """Approximate token count."""
        return len(text.split())
    
    def create_chunks(self, laptop_data: Dict[str, Any]) -> List[LaptopChunk]:
        """Create field-base chunks."""
        chunks = []
        laptop_id = laptop_data["laptop_id"]
        
        for field in self.important_fields:
            value = laptop_data.get(field)
            
            # Skip empty/null values
            if pd.isna(value) or value == "" or value is None:
                continue
            
            # Create chunk content with context
            field_name_readable = field.replace("_", " ").title()
            producer = laptop_data.get("producer", "")
            model = laptop_data.get("model", "")
            
            # Add context for better retrieval
            if field in ["producer", "model", "full_name"]:
                content = f"{value}"
            else:
                content = f"{producer} {model} - {field_name_readable}: {value}"
            
            token_count = self._count_tokens(content)
            
            chunk = LaptopChunk(
                chunk_id=f"{laptop_id}:{field}",
                laptop_id=laptop_id,
                content=content,
                metadata={
                    "field": field,
                    "producer": producer,
                    "model": model,
                    "full_name": laptop_data.get("full_name", "")
                },
                token_count=token_count,
                field_name=field
            )
            chunks.append(chunk)
        
        return chunks


class LaptopDataIngestion:
    """
    Ingestion pipeline for laptop dataset.
    
    Design Pattern: Pipeline with transformation stages
    """
    
    def __init__(self, csv_path: str, chunking_strategy: ChunkingStrategy):
        self.csv_path = csv_path
        self.chunking_strategy = chunking_strategy
        self.df: Optional[pd.DataFrame] = None
        self.chunks: List[LaptopChunk] = []

    def _require_dataframe(self) -> pd.DataFrame:
        """Return loaded dataframe or raise if load_data has not been run."""
        if self.df is None:
            raise RuntimeError("Dataset not loaded. Call load_data() before processing.")
        return self.df
    
    def load_data(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load and optionally sample dataset."""
        print(f"Loading dataset from {self.csv_path}...")
        self.df = pd.read_csv(self.csv_path)
        
        if sample_size and sample_size < len(self.df):
            print(f"Sampling {sample_size} laptops for faster processing...")
            self.df = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        print(f"Loaded {len(self.df)} laptops")
        return self.df
    
    def normalize_data(self) -> pd.DataFrame:
        """Apply data normalizations."""
        print("Normalizing data...")
        df = self._require_dataframe()
        
        # Strip whitespace from string columns
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        
        # Normalize price column
        if 'price_in_dollar' in df.columns:
            df['price_in_dollar'] = df['price_in_dollar'].astype(str).str.replace('$', '').str.replace(',', '')
        
        # Normalize CPU/GPU names
        for col in ['cpu', 'gpu']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(r'\s+', ' ', regex=True)
        
        self.df = df
        return df
    
    def create_chunks(self) -> List[LaptopChunk]:
        """Create chunks from all laptops."""
        print("Creating chunks...")
        df = self._require_dataframe()
        self.chunks = []
        
        for idx, row in df.iterrows():
            laptop_data = row.to_dict()
            laptop_chunks = self.chunking_strategy.create_chunks(laptop_data)
            self.chunks.extend(laptop_chunks)
        
        print(f"Created {len(self.chunks)} chunks from {len(df)} laptops")
        return self.chunks
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        if not self.chunks:
            return {}
        
        token_counts = [c.token_count for c in self.chunks]
        df = self._require_dataframe()
        return {
            "total_laptops": len(df),
            "total_chunks": len(self.chunks),
            "avg_chunks_per_laptop": len(self.chunks) / len(df),
            "avg_tokens_per_chunk": np.mean(token_counts),
            "min_tokens": int(np.min(token_counts)),
            "max_tokens": int(np.max(token_counts))
        }


# ============================================================================
# RETRIEVAL
# ============================================================================

class RetrievalStrategy(ABC):
    """Abstract base for retrieval strategies."""
    
    @abstractmethod
    def index(self, chunks: List[LaptopChunk]) -> None:
        """Index chunks for retrieval."""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[LaptopChunk], List[float]]:
        """Retrieve top-k chunks for query."""
        pass

    @property
    def name(self) -> str:
        """Human-readable name used for logging and evaluation."""
        return self.__class__.__name__


class BM25Retrieval(RetrievalStrategy):
    """
    BM25 retrieval strategy.
    
    Why: Fast, interpretable, no GPU needed; works well for keyword matching.
    How: Tokenize corpus, build inverted index, rank by BM25 scores.
    """
    
    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[LaptopChunk] = []
        self.tokenized_corpus: List[List[str]] = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization with lowercasing."""
        return text.lower().split()
    
    def index(self, chunks: List[LaptopChunk]) -> None:
        """Build BM25 index."""
        print(f"Building BM25 index for {len(chunks)} chunks...")
        self.chunks = chunks
        self.tokenized_corpus = [self._tokenize(chunk.content) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print("BM25 index built successfully")
    
    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[LaptopChunk], List[float]]:
        """Retrieve top-k chunks using BM25."""
        if not self.bm25:
            raise RuntimeError("Index not built. Call index() first.")
        
        with mlflow.start_span(name="retrieval") as span:
            span.set_inputs({
                "query": query,
                "top_k": top_k,
                "retrieval_method": "BM25"
            })
            
            tokenized_query = self._tokenize(query)
            scores = self.bm25.get_scores(tokenized_query)
            
            # Get top-k indices
            top_k_indices = np.argsort(scores)[::-1][:top_k]
            
            top_chunks = [self.chunks[i] for i in top_k_indices]
            top_scores = [float(scores[i]) for i in top_k_indices]
            
            span.set_outputs({
                "num_results": len(top_chunks),
                "top_scores": top_scores[:3],
                "chunk_ids": [f"{c.laptop_id}:{c.field_name}" for c in top_chunks]
            })
            
            span.set_attributes({
                "retrieval_method": "BM25",
                "corpus_size": len(self.chunks),
                "avg_score": float(np.mean(top_scores)) if top_scores else 0.0
            })
            
            return top_chunks, top_scores


# ============================================================================
# GENERATION
# ============================================================================

class AnswerGenerator:
    """
    Generates answers with citations from retrieved chunks.
    
    Design Pattern: Template Method with LLM integration
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo-0125", temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature
        self.llm: BaseChatModel = ChatOpenAI(model=model_name, temperature=temperature)  # type: ignore[call-arg]
        # self.llm: BaseChatModel = ChatOllama(model=model_name, temperature=temperature)  # type: ignore[call-arg]
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert laptop technical advisor. Answer questions based ONLY on the provided laptop specifications.

CRITICAL RULES:
1. Use ONLY information from the provided chunks
2. ALWAYS cite sources using format [laptop_id:field_name]
3. Keep answers ≤120 words
4. Be precise and technical
5. If information is not in chunks, say "Information not available"

Example:
Query: "What CPU does the ASUS VivoBook have?"
Chunks: 
- [1:cpu] "ASUS VivoBook - CPU: Intel Core i7-1260P"
Answer: "The ASUS VivoBook features an Intel Core i7-1260P processor [1:cpu]."
"""),
            ("user", """Question: {query}

Available Laptop Specifications:
{chunks}

Provide a concise answer (≤120 words) with citations in format [laptop_id:field_name].""")
        ])
        
        self.chain = self.prompt_template | self.llm | StrOutputParser()
    
    def generate(self, query: str, chunks: List[LaptopChunk]) -> GenerationResult:
        """Generate answer with citations."""
        # Format chunks for prompt
        chunks_text = "\n".join([
            f"[{chunk.laptop_id}:{chunk.field_name}] {chunk.content}"
            for chunk in chunks
        ])
        
        with mlflow.start_span(name="answer_generation") as span:
            span.set_inputs({
                "query": query,
                "chunks": chunks_text,
                "num_chunks": len(chunks)
            })
            
            # Generate answer
            answer = self.chain.invoke({
                "query": query,
                "chunks": chunks_text
            })
            
            # Extract citations
            citation_pattern = r'\[(\d+:[a-z_]+)\]'
            citations = re.findall(citation_pattern, answer)
            
            # Count words
            word_count = len(answer.split())
            
            # Track which chunks were cited
            used_chunk_ids = list(set(citations))
            
            span.set_outputs({
                "answer": answer,
                "citations": citations,
                "word_count": word_count,
                "used_chunks": used_chunk_ids
            })
            
            span.set_attributes({
                "model": self.model_name,
                "temperature": self.temperature,
                "num_citations": len(citations),
                "unique_citations": len(used_chunk_ids)
            })
            
            return GenerationResult(
                answer=answer,
                citations=citations,
                used_chunks=used_chunk_ids,
                word_count=word_count
            )


# ============================================================================
# CRITICAL AGENT (LangGraph)
# ============================================================================

class CriticalAgent:
    """
    Critical agent that validates claims against retrieved passages.
    
    Design Pattern: Chain of Responsibility via LangGraph
    Why: Ensures factual accuracy; prevents hallucinations
    How: Decomposes answer into claims, checks each against chunks, regenerates if needed
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo-0125", max_iterations: int = 2):
        self.model_name = model_name
        self.max_iterations = max_iterations
        # self.llm = ChatOllama(model=model_name, temperature=0.0, format='json')  # type: ignore[call-arg]
        self.llm = ChatOpenAI(model=model_name, temperature=0.0, response_format={"type": "json_object"})  # type: ignore[call-arg]
        
        # Generator for regeneration within the agent
        self.regenerate_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert laptop technical advisor. Your previous answer had issues and needs correction.

CRITICAL RULES:
1. Use ONLY information from the provided chunks
2. ALWAYS cite sources using format [laptop_id:field_name]
3. Keep answers ≤120 words
4. Be precise and technical
5. Only make claims that can be directly verified in the chunks
6. If information is not in chunks, say "Information not available"
7. If the unsupported claims state "Information not available", do not attempt to regenerate; accept the answer as is.

PREVIOUS ANSWER (REJECTED):
{previous_answer}

VALIDATION RESULTS:
✅ SUPPORTED CLAIMS (keep these):
{supported_claims}

❌ UNSUPPORTED CLAIMS (fix or remove these):
{unsupported_claims}

Learn from these mistakes and generate a corrected answer that only includes verifiable claims with proper citations.
"""),
            ("user", """Question: {query}

Available Laptop Specifications:
{chunks}

Generate a NEW, CORRECTED answer (≤120 words) with proper citations [laptop_id:field_name] in json format.""")
        ])
        
        # Build LangGraph workflow
        self.graph = self._build_graph()
    
    def _build_graph(self) -> Any:
        """Build LangGraph state machine for critical agent."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("validate_claims", self._validate_claims_node)
        workflow.add_node("decide_action", self._decide_action_node)
        workflow.add_node("regenerate_answer", self._regenerate_answer_node)  # NEW: Regeneration node
        workflow.add_node("finalize", self._finalize_node)
        
        # Add edges
        workflow.set_entry_point("validate_claims")
        workflow.add_edge("validate_claims", "decide_action")
        
        # Conditional edge based on validation result
        workflow.add_conditional_edges(
            "decide_action",
            self._should_regenerate,
            {
                "regenerate": "regenerate_answer",  # FIXED: Go to regeneration node
                "accept": "finalize",
                "reject": "finalize"
            }
        )
        
        # After regeneration, validate again
        workflow.add_edge("regenerate_answer", "validate_claims")
        
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _validate_claims_node(self, state: AgentState) -> AgentState:
        """Validate claims against retrieved chunks."""
        print(f"\n[Critical Agent] Iteration {state['iteration'] + 1}")
        
        answer = state["generated_answer"]
        chunks = state["retrieved_chunks"]
        
        # Create chunks text for validation
        chunks_text = "\n".join([
            f"[{c['laptop_id']}:{c['field_name']}] {c['content']}"
            for c in chunks
        ])
        
        validation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a fact-checking agent. Your job is to verify if claims in an answer are supported by the provided evidence.

TASK:
1. Break down the answer into individual claims
2. For each claim, check if it's supported by the chunks
3. Classify as SUPPORTED or UNSUPPORTED

Output JSON format:
{{
  "supported_claims": ["claim 1", "claim 2"],
  "unsupported_claims": ["claim 3"],
  "is_valid": true/false
}}

A claim is SUPPORTED if the exact information appears in the chunks with proper citation.
A claim is UNSUPPORTED if it's not in the chunks or lacks citation."""),
            ("user", """Answer to validate:
{answer}

Evidence chunks:
{chunks}

Validate and return JSON with supported_claims, unsupported_claims, and is_valid fields.""")
        ])
        
        validation_chain = validation_prompt | self.llm | StrOutputParser()
        
        with mlflow.start_span(name="critical_agent_validation") as span:
            span.set_inputs({
                "answer": answer,
                "chunks": chunks_text,
                "iteration": state["iteration"]
            })
            
            response = validation_chain.invoke({
                "answer": answer,
                "chunks": chunks_text
            })
            
            # Parse JSON response
            try:
                result = json.loads(response)
                state["supported_claims"] = result.get("supported_claims", [])
                state["unsupported_claims"] = result.get("unsupported_claims", [])
                state["is_valid"] = result.get("is_valid", False)
            except json.JSONDecodeError:
                # Fallback: assume invalid if can't parse
                state["supported_claims"] = []
                state["unsupported_claims"] = ["Could not parse validation response"]
                state["is_valid"] = False
            
            span.set_outputs({
                "llm_response": response,
                "supported_claims": state["supported_claims"],
                "unsupported_claims": state["unsupported_claims"],
                "is_valid": state["is_valid"]
            })
            
            span.set_attributes({
                "model": self.model_name,
                "iteration": state["iteration"],
                "num_supported": len(state["supported_claims"]),
                "num_unsupported": len(state["unsupported_claims"])
            })
        
        return state
    
    def _decide_action_node(self, state: AgentState) -> AgentState:
        """Decide what action to take based on validation."""
        is_valid = state["is_valid"]
        iteration = state["iteration"]
        max_iterations = state["max_iterations"]
        
        if is_valid:
            state["action"] = "accept"
            state["decision_reason"] = "All claims are supported by retrieved chunks"
        elif iteration >= max_iterations - 1:
            state["action"] = "reject"
            state["decision_reason"] = f"Max iterations ({max_iterations}) reached with unsupported claims"
        else:
            state["action"] = "regenerate"
            state["decision_reason"] = "Unsupported claims found, regenerating..."
            state["iteration"] = iteration + 1
        
        # Log decision
        decision_log = {
            "iteration": iteration,
            "action": state["action"],
            "reason": state["decision_reason"],
            "unsupported_claims": state["unsupported_claims"],
            "timestamp": datetime.now().isoformat()
        }
        state["decisions_log"].append(decision_log)
        
        # MLflow tracking
        with mlflow.start_span(name="critical_agent_decision") as span:
            span.set_inputs({
                "is_valid": is_valid,
                "iteration": iteration,
                "max_iterations": max_iterations
            })
            
            span.set_outputs({
                "action": state["action"],
                "reason": state["decision_reason"]
            })
            
            span.set_attributes({
                "iteration": iteration,
                "num_unsupported_claims": len(state["unsupported_claims"])
            })
        
        print(f"[Critical Agent] Action: {state['action']}")
        print(f"[Critical Agent] Reason: {state['decision_reason']}")
        
        return state
    
    def _regenerate_answer_node(self, state: AgentState) -> AgentState:
        """Regenerate answer based on validation feedback."""
        print(f"\n[Critical Agent] Regenerating answer (attempt {state['iteration'] + 1})...")
        
        query = state["query"]
        chunks = state["retrieved_chunks"]
        previous_answer = state["generated_answer"]
        unsupported_claims = state["unsupported_claims"]
        supported_claims = state["supported_claims"]
        
        # Format chunks for prompt
        chunks_text = "\n".join([
            f"[{c['laptop_id']}:{c['field_name']}] {c['content']}"
            for c in chunks
        ])
        
        # Format claims as feedback
        supported_text = "\n".join([f"- {claim}" for claim in supported_claims]) if supported_claims else "None"
        unsupported_text = "\n".join([f"- {claim}" for claim in unsupported_claims]) if unsupported_claims else "None"
        
        # Generate new answer with feedback
        regeneration_chain = self.regenerate_prompt | self.llm | StrOutputParser()
        
        with mlflow.start_span(name="critical_agent_regeneration") as span:
            span.set_inputs({
                "query": query,
                "previous_answer": previous_answer,
                "chunks": chunks_text,
                "supported_claims": supported_claims,
                "unsupported_claims": unsupported_claims,
                "iteration": state["iteration"]
            })
            
            new_answer = regeneration_chain.invoke({
                "query": query,
                "chunks": chunks_text,
                "previous_answer": previous_answer,
                "supported_claims": supported_text,
                "unsupported_claims": unsupported_text
            })
            
            # Update state with new answer
            state["generated_answer"] = new_answer
            
            # Extract new citations
            citation_pattern = r'\[(\d+:[a-z_]+)\]'
            new_citations = re.findall(citation_pattern, new_answer)
            state["citations"] = new_citations
            
            span.set_outputs({
                "new_answer": new_answer,
                "new_citations": new_citations,
                "previous_answer": previous_answer
            })
            
            span.set_attributes({
                "model": self.model_name,
                "iteration": state["iteration"],
                "num_new_citations": len(new_citations),
                "num_supported_claims_from_previous": len(supported_claims),
                "num_unsupported_claims_from_previous": len(unsupported_claims)
            })
        
        print(f"[Critical Agent] Generated new answer with {len(new_citations)} citations")
        print(f"[Critical Agent] Previous answer had {len(supported_claims)} supported and {len(unsupported_claims)} unsupported claims")
        
        return state
    
    def _finalize_node(self, state: AgentState) -> AgentState:
        """Finalize the answer."""
        if state["action"] == "accept":
            state["final_answer"] = state["generated_answer"]
        else:
            state["final_answer"] = "Unable to generate a fully supported answer from available data."
        
        return state
    
    def _should_regenerate(self, state: AgentState) -> str:
        """Decide whether to regenerate, accept, or reject."""
        return state["action"]
    
    def validate(
        self, 
        query: str,
        answer: str,
        citations: List[str],
        chunks: List[LaptopChunk]
    ) -> CriticalAgentDecision:
        """Run validation through LangGraph."""
        # Initialize state
        initial_state: AgentState = {
            "query": query,
            "retrieved_chunks": [c.to_dict() for c in chunks],
            "retrieval_scores": [],
            "generated_answer": answer,
            "citations": citations,
            "is_valid": False,
            "unsupported_claims": [],
            "supported_claims": [],
            "decision_reason": "",
            "action": "",
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "final_answer": "",
            "decisions_log": []
        }
        
        # Run graph
        final_state = self.graph.invoke(initial_state)
        
        # Create decision object
        decision = CriticalAgentDecision(
            is_valid=final_state["is_valid"],
            unsupported_claims=final_state["unsupported_claims"],
            supported_claims=final_state["supported_claims"],
            reason=final_state["decision_reason"],
            action=final_state["action"]
        )
        
        return decision


# ============================================================================
# RAG PIPELINE
# ============================================================================

class RAGPipeline:
    """
    Complete RAG pipeline with critical agent.
    
    Design Pattern: Facade + Dependency Injection
    """
    
    def __init__(
        self,
        retrieval_strategy: RetrievalStrategy,
        generator: AnswerGenerator,
        critical_agent: CriticalAgent,
        top_k: int = 5
    ):
        self.retrieval = retrieval_strategy
        self.generator = generator
        self.critical_agent = critical_agent
        self.top_k = top_k
        self.retrieval_method = getattr(retrieval_strategy, "name", retrieval_strategy.__class__.__name__)
    
    def query(self, query: str) -> RAGResult:
        """Execute complete RAG pipeline with critical agent."""
        with mlflow.start_span(name="rag_pipeline") as span:
            start_time = time.time()
            
            span.set_inputs({
                "query": query,
                "top_k": self.top_k,
                "retrieval_method": self.retrieval_method
            })
            
            # Step 1: Retrieve
            print(f"\n[RAG Pipeline] Query: {query}")
            print("[RAG Pipeline] Step 1: Retrieval...")
            chunks, scores = self.retrieval.retrieve(query, top_k=self.top_k)
            
            retrieval_result = RetrievalResult(
                query=query,
                chunks=chunks,
                scores=scores,
                retrieval_method=self.retrieval_method
            )
            
            print(f"[RAG Pipeline] Retrieved {len(chunks)} chunks")
            
            # Step 2: Generate
            print("[RAG Pipeline] Step 2: Generation...")
            generation_result = self.generator.generate(query, chunks)
            
            print(f"[RAG Pipeline] Generated answer ({generation_result.word_count} words)")
            print(f"[RAG Pipeline] Citations: {generation_result.citations}")
            
            # Step 3: Critical Agent Validation
            print("[RAG Pipeline] Step 3: Critical Agent Validation...")
            decision = self.critical_agent.validate(
                query=query,
                answer=generation_result.answer,
                citations=generation_result.citations,
                chunks=chunks
            )
            
            # Determine final answer based on decision
            if decision.action == "accept":
                final_answer = generation_result.answer
            else:
                final_answer = "Unable to provide a fully supported answer from available specifications."
            
            total_latency = time.time() - start_time
            
            result = RAGResult(
                query=query,
                retrieval=retrieval_result,
                generation=generation_result,
                critical_decision=decision,
                final_answer=final_answer,
                iteration_count=1,  # Simplified for now
                total_latency=total_latency
            )
            
            # Log pipeline outputs
            span.set_outputs({
                "final_answer": final_answer,
                "decision_action": decision.action,
                "num_chunks_retrieved": len(chunks),
                "num_citations": len(generation_result.citations),
                "total_latency": total_latency
            })
            
            span.set_attributes({
                "retrieval_method": self.retrieval_method,
                "top_k": self.top_k,
                "word_count": generation_result.word_count,
                "is_valid": decision.is_valid,
                "num_supported_claims": len(decision.supported_claims),
                "num_unsupported_claims": len(decision.unsupported_claims)
            })
            
            print(f"\n[RAG Pipeline] Final Decision: {decision.action}")
            print(f"[RAG Pipeline] Total Latency: {total_latency:.2f}s")
            
            return result


# ============================================================================
# EVALUATION
# ============================================================================

@dataclass
class EvaluationMetrics:
    """Evaluation metrics for RAG system."""
    precision_at_k: float
    recall_at_k: float
    faithfulness: float  # % supported sentences
    answer_coverage: float  # % queries with answers
    avg_latency: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "faithfulness": self.faithfulness,
            "answer_coverage": self.answer_coverage,
            "avg_latency": self.avg_latency
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution pipeline with CLI argument parsing."""
    
    # Load environment variables
    load_dotenv()
    
    # Verify OpenAI API key is loaded
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please create a .env file with your API key.")
    
    # Parse command line arguments
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="RAG + Critical Agent System for Laptop QA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data & Ingestion
    parser.add_argument(
        "--csv-path",
        type=str,
        default="../data/Laptops_with_technical_specifications.csv",
        help="Path to laptop specifications CSV file"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=300,
        help="Number of laptops to sample (None = all)"
    )
    
    # Retrieval Configuration
    parser.add_argument(
        "--retrieval-backend",
        type=str,
        choices=["bm25", "langchain_vector"],
        default="bm25",
        help="Retrieval strategy to use"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per query"
    )
    parser.add_argument(
        "--vector-store-dir",
        type=str,
        default="./vector_index",
        help="Directory for vector store persistence"
    )
    
    # Model Configuration
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt-3.5-turbo-0125",
        help="OpenAI model name for generation and validation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature for generation"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum regeneration iterations for critical agent"
    )
    
    # Chunking Configuration
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=10,
        help="Minimum tokens per chunk"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=120,
        help="Maximum tokens per chunk"
    )
    
    # MLflow Configuration
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow tracking server URI"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="RAG_Critical_Agent_Laptop_QA",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="RAG_Test_Queries",
        help="MLflow run name"
    )
    
    # Execution Configuration
    parser.add_argument(
        "--output-path",
        type=str,
        default="rag_critical_agent_log.json",
        help="Path to save results JSON"
    )
    
    args: argparse.Namespace = parser.parse_args()
    
    # Display configuration banner
    print("=" * 70)
    print("RAG + CRITICAL AGENT SYSTEM FOR LAPTOP QA")
    print("=" * 70)
    print(f"\n📋 Configuration:")
    print(f"  Dataset: {args.csv_path}")
    print(f"  Sample Size: {args.sample_size}")
    print(f"  Retrieval: {args.retrieval_backend.upper()} (top-k={args.top_k})")
    print(f"  Model: {args.model_name} (temp={args.temperature})")
    print(f"  Critical Agent: max_iterations={args.max_iterations}")
    print(f"  MLflow URI: {args.mlflow_uri}")
    print(f"  Experiment: {args.experiment_name}")
    
    # MLflow Setup
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.langchain.autolog()
    
    experiment: Optional[Any] = mlflow.get_experiment_by_name(args.experiment_name)
    if experiment is None:
        experiment_id: str = mlflow.create_experiment(
            args.experiment_name,
            tags={"project": "laptop_qa", "system": "rag_critical_agent"}
        )
    else:
        experiment_id: str = experiment.experiment_id
    
    mlflow.set_experiment(args.experiment_name)
    print(f"\n[MLflow] Experiment ID: {experiment_id}")
    
    # ========================================================================
    # STEP 1: INGESTION & CHUNKING
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: INGESTION & CHUNKING")
    print("=" * 70)
    
    chunking_strategy: ChunkingStrategy = FieldBasedChunking(
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens
    )
    ingestion: LaptopDataIngestion = LaptopDataIngestion(args.csv_path, chunking_strategy)
    
    df: pd.DataFrame = ingestion.load_data(sample_size=args.sample_size if args.sample_size > 0 else None)
    df = ingestion.normalize_data()
    chunks: List[LaptopChunk] = ingestion.create_chunks()
    
    stats: Dict[str, Any] = ingestion.get_statistics()
    print(f"\n📊 Ingestion Statistics:")
    for key, value in stats.items():
        print(f"  • {key}: {value}")
    
    # ========================================================================
    # STEP 2: INDEXING
    # ========================================================================
    print("\n" + "=" * 70)
    print(f"STEP 2: INDEXING ({args.retrieval_backend.upper()})")
    print("=" * 70)
    
    retrieval_strategy: RetrievalStrategy = BM25Retrieval()
    retrieval_strategy.index(chunks)
    
    # ========================================================================
    # STEP 3: INITIALIZE COMPONENTS
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: INITIALIZE COMPONENTS")
    print("=" * 70)
    
    generator: AnswerGenerator = AnswerGenerator(
        model_name=args.model_name,
        temperature=args.temperature
    )
    critical_agent: CriticalAgent = CriticalAgent(
        model_name=args.model_name,
        max_iterations=args.max_iterations
    )
    
    pipeline: RAGPipeline = RAGPipeline(
        retrieval_strategy=retrieval_strategy,
        generator=generator,
        critical_agent=critical_agent,
        top_k=args.top_k
    )
    
    print("✅ Components initialized successfully")
    
    # ========================================================================
    # STEP 4: EXECUTE TEST QUERIES
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: EXECUTE TEST QUERIES")
    print("=" * 70)
    
    # Define test queries with ground truth
    TEST_QUERIES: Final[List[Dict[str, Any]]] = [
        {
            "query": "What laptops have Intel Core i7 processors?",
        },
        {
            "query": "Which laptops have NVIDIA RTX 3050 GPU?",
        },
        {
            "query": "What are the display specifications of ASUS laptops?",
        },
        {
            "query": "List laptops with at least 16GB RAM and 512GB SSD.",
        },
        {
            "query": "Which Dell laptops feature a an DIAMOND-V0 GPU?"
        }
    ]
    
    # Execute queries within MLflow run
    with mlflow.start_run(run_name=args.run_name) as run:
        print(f"\n[MLflow] Run ID: {run.info.run_id}")
        
        # Log configuration parameters
        mlflow.log_params({
            "sample_size": args.sample_size,
            "top_k": args.top_k,
            "model": args.model_name,
            "temperature": args.temperature,
            "retrieval_backend": args.retrieval_backend,
            "chunking_strategy": "FieldBasedChunking",
            "min_tokens": args.min_tokens,
            "max_tokens": args.max_tokens,
            "max_iterations": args.max_iterations,
            "num_queries": len(TEST_QUERIES)
        })
        
        test_results: List[RAGResult] = []
        for i, test_case in enumerate(TEST_QUERIES, start=1):
            print(f"\n[Query {i}/{len(TEST_QUERIES)}] {test_case['query']}")
            result: RAGResult = pipeline.query(test_case['query'])
            test_results.append(result)
            
            print(f"  💬 Answer: {result.final_answer}")
            print(f"  ✓ Validation: {result.critical_decision.action}")
            print(f"  ⏱️  Latency: {result.total_latency:.2f}s")
            print("-" * 70)
        
        # Compute and log aggregate metrics
        avg_latency: float = sum(r.total_latency for r in test_results) / len(test_results)
        accepted_count: int = sum(1 for r in test_results if r.critical_decision.action == "accept")
        acceptance_rate: float = accepted_count / len(test_results)
        
        mlflow.log_metrics({
            "avg_latency": avg_latency,
            "acceptance_rate": acceptance_rate,
            "total_queries": len(test_results),
            "accepted_queries": accepted_count
        })
        
        print(f"\n📈 Aggregate Metrics:")
        print(f"  • Average Latency: {avg_latency:.2f}s")
        print(f"  • Acceptance Rate: {acceptance_rate:.1%}")
        print(f"  • Accepted Queries: {accepted_count}/{len(test_results)}")
        
        # ====================================================================
        # STEP 5: SAVE RESULTS
        # ====================================================================
        print("\n" + "=" * 70)
        print("STEP 5: SAVE RESULTS")
        print("=" * 70)
        
        results_payload: Dict[str, Any] = {
            "test_results": [r.to_dict() for r in test_results],
            "configuration": {
                "sample_size": args.sample_size,
                "top_k": args.top_k,
                "model": args.model_name,
                "temperature": args.temperature,
                "chunking_strategy": "FieldBasedChunking",
                "retrieval_backend": args.retrieval_backend,
                "min_tokens": args.min_tokens,
                "max_tokens": args.max_tokens,
                "max_iterations": args.max_iterations
            },
            "aggregate_metrics": {
                "avg_latency": avg_latency,
                "acceptance_rate": acceptance_rate,
                "total_queries": len(test_results),
                "accepted_queries": accepted_count
            },
            "timestamp": datetime.now().isoformat()
        }
        
        output_path: Path = Path(args.output_path)
        with open(output_path, "w") as f:
            json.dump(results_payload, f, indent=2)
        
        mlflow.log_artifact(str(output_path))
        
        print(f"✅ Results saved to: {output_path}")
        print(f"[MLflow] View traces at: {mlflow.get_tracking_uri()}")
        print(f"[MLflow] Run ID: {run.info.run_id}")
    
    print("\n🎉 Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
