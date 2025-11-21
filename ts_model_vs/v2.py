"""
Production-grade anomaly detection framework with bootstrap A/B testing.

This module provides a flexible architecture for comparing anomaly detection models
using bootstrap F1-delta methodology with proper statistical testing.
"""

from typing import TypedDict
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from langchain_core.language_models import BaseChatModel
from langchain_openai.chat_models import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
import asyncio
from dotenv import load_dotenv
import os
import mlflow
import mlflow.langchain  # type: ignore
import argparse

@dataclass
class ModelMetadata:
    """Stores training metadata computed during fit phase."""
    mean: float = 0.0
    std: float = 0.0
    median: float = 0.0
    q25: float = 0.0
    q75: float = 0.0
    n_samples: int = 0
    feature_range: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    item_id: Optional[str] = None
    
    def __repr__(self) -> str:
        item_str: str = f", item={self.item_id}" if self.item_id else ""
        return (
            f"ModelMetadata(mean={self.mean:.2f}, std={self.std:.2f}, "
            f"median={self.median:.2f}, n_samples={self.n_samples}{item_str})"
        )

class BaseAnomalyModel(ABC):
    """Abstract base for all anomaly detection models."""
    
    def __init__(self) -> None:
        self.metadata: Dict[str, ModelMetadata] = {}
        self._is_fitted: bool = False
    
    def fit(self, X: np.ndarray, item_id: Optional[str] = None) -> 'BaseAnomalyModel':
        """Compute and store training data statistics per item."""
        X_flat: np.ndarray = self._ensure_1d(X)
        key: str = item_id if item_id is not None else "global"
        
        self.metadata[key] = ModelMetadata(
            mean=float(np.mean(X_flat)),
            std=float(np.std(X_flat)),
            median=float(np.median(X_flat)),
            q25=float(np.percentile(X_flat, 25)),
            q75=float(np.percentile(X_flat, 75)),
            n_samples=len(X_flat),
            feature_range=(float(np.min(X_flat)), float(np.max(X_flat))),
            item_id=item_id
        )
        
        self._is_fitted = True
        return self
    
    def score(self, X: np.ndarray, item_id: Optional[str] = None) -> np.ndarray:
        """Validate that model is fitted and data is properly formatted."""
        if not self._is_fitted:
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before scoring")
        
        return self._ensure_1d(X)
    
    @abstractmethod
    def _compute_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Model-specific scoring logic."""
        pass
    
    def _ensure_1d(self, X: np.ndarray) -> np.ndarray:
        """Convert input to 1D array."""
        if isinstance(X, pd.Series):
            return X.values
        
        X_array: np.ndarray = np.asarray(X)
        
        if X_array.ndim == 2 and X_array.shape[1] == 1:
            return X_array.ravel()
        elif X_array.ndim == 1:
            return X_array
        else:
            raise ValueError(f"Expected 1D or (n, 1) array, got shape {X_array.shape}")

class ItemData(TypedDict):
        """Type definition for item-level detection data."""
        item_id: str
        prices: np.ndarray
        timestamps: np.ndarray
        scores: np.ndarray
        labels: np.ndarray
        reasons: Optional[List[str]]
    
class ModelData(TypedDict):
    """Type definition for model-level data structure."""
    model_object: BaseAnomalyModel
    name: str
    items: List[ItemData]
    
class ZScoreModel(BaseAnomalyModel):
    """Anomaly detection using statistical Z-scores."""
    
    def __init__(self, threshold: float = 3.0) -> None:
        super().__init__()
        self.threshold: float = threshold
    
    def fit(self, X: np.ndarray, item_id: Optional[str] = None) -> 'ZScoreModel':
        super().fit(X, item_id)
        return self
    
    def score(self, X: np.ndarray, item_id: Optional[str] = None) -> np.ndarray:
        X_validated: np.ndarray = super().score(X, item_id)
        key: str = item_id if item_id is not None else "global"
        return self._compute_anomaly_scores(X_validated, key)
    
    def _compute_anomaly_scores(self, X: np.ndarray, key: str) -> np.ndarray:
        meta: ModelMetadata = self.metadata[key]
        z_scores: np.ndarray = np.abs((X - meta.mean) / (meta.std + 1e-10))
        return z_scores


class MovingQuantileModel(BaseAnomalyModel):
    """Anomaly detection using rolling quantile deviations."""
    
    def __init__(self, window_size: int = 10, quantile: float = 0.95) -> None:
        super().__init__()
        self.window_size: int = window_size
        self.quantile: float = quantile
    
    def fit(self, X: np.ndarray, item_id: Optional[str] = None) -> 'MovingQuantileModel':
        super().fit(X, item_id)
        return self
    
    def score(self, X: np.ndarray, item_id: Optional[str] = None) -> np.ndarray:
        X_validated: np.ndarray = super().score(X, item_id)
        key: str = item_id if item_id is not None else "global"
        return self._compute_anomaly_scores(X_validated, key)
    
    def _compute_anomaly_scores(self, X: np.ndarray, key: str) -> np.ndarray:
        meta: ModelMetadata = self.metadata[key]
        iqr: float = meta.q75 - meta.q25
        
        lower_bound: float = meta.q25 - 1.5 * iqr
        upper_bound: float = meta.q75 + 1.5 * iqr
        
        below_lower: np.ndarray = np.maximum(0, lower_bound - X)
        above_upper: np.ndarray = np.maximum(0, X - upper_bound)
        
        raw_scores: np.ndarray = below_lower + above_upper
        scores: np.ndarray = raw_scores / (iqr + 1e-10)
        
        return scores

class NewLLMAnomalyDetector(BaseAnomalyModel):
    """
    Concurrent LLM-based anomaly detection using LangChain async processing.
    
    Design Pattern: Map-Reduce with async batching
    
    Why: Processes all items concurrently for massive performance improvement.
    Traditional approach processes 300 items sequentially (hours), this processes
    them in parallel (minutes).
    
    How: Uses RunnableMap to fit metadata extraction and LLM scoring in parallel.
    Each item gets its own chain that extracts statistics (fit) and scores anomalies.
    """
    _autolog_configured: bool = False
    
    def __init__(
        self,
        base_model: BaseChatModel,
        window_size: int = 10,
        model_name: str = "gpt-3.5-turbo-0125",
        temperature: float = 0.0,
        max_concurrent: int = 10
    ) -> None:
        """
        Initialize concurrent LLM-based anomaly detector.
        
        Args:
            window_size: Number of historical points to include in context
            model_name: Ollama model name
            temperature: LLM temperature (0.0 for deterministic)
            max_concurrent: Maximum concurrent LLM requests
        """
        super().__init__()
        self.window_size: int = window_size
        self.model_name: str = model_name
        self.temperature: float = temperature
        self.max_concurrent: int = max_concurrent
        self._llm: BaseChatModel = base_model
        self.total_latency: float = 0.0
        self.total_tokens: int = 0
        self.total_llm_calls: int = 0
        self.detailed_results: Optional[pd.DataFrame] = None
        if not NewLLMAnomalyDetector._autolog_configured:
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.langchain.autolog()  # type: ignore[attr-defined]
            NewLLMAnomalyDetector._autolog_configured = True
    
    def fit(self, X: np.ndarray, item_id: Optional[str]) -> Dict[str, Any]:
        """
        Extract metadata from item data (equivalent to 'fit' per item).
        
        This is called as a RunnableLambda before LLM invocation.
        Runs in parallel for all items.
        
        """
        super().fit(X, item_id)

    def _extract_item_metadata(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from item data."""
        
        item_id: str = item_data["item_id"]
        prices: np.ndarray = np.array(item_data["prices"])
        
        # Use precomputed global stats if available (for chunks), otherwise compute local
        if "precomputed_stats" in item_data:
            stats = item_data["precomputed_stats"]
            metadata = ModelMetadata(
                mean=stats["mean"],
                std=stats["std"],
                median=stats["median"],
                q25=stats["q25"],
                q75=stats["q75"],
                n_samples=stats["n_samples"],
                feature_range=(stats["min"], stats["max"]),
                item_id=item_data.get("original_item_id", item_id)
            )
        else:
            metadata = ModelMetadata(
                mean=float(np.mean(prices)),
                std=float(np.std(prices)),
                median=float(np.median(prices)),
                q25=float(np.percentile(prices, 25)),
                q75=float(np.percentile(prices, 75)),
                n_samples=len(prices),
                feature_range=(float(np.min(prices)), float(np.max(prices))),
                item_id=item_id
            )
        
        self.metadata[item_id] = metadata

        return {
            "item_id": item_id,
            "prices": prices,
            "metadata": metadata,
            "ts_list": item_data.get("ts_list", ['NO TIMESTAMPS'] * len(prices))
        }
    


    
    def _create_item_chain(self):
        """
        Create the processing chain for a single item.
        
        Chain structure:
        1. RunnableMap: Extracts metadata in parallel with passthrough
        2. Prompt: Creates LLM prompt with metadata and prices
        3. LLM: Scores anomalies
        4. Parser: Extracts scores from response
        """
        from langchain_core.runnables import RunnableLambda, RunnableMap
        
        # Step 1: Metadata extraction lambda
        metadata_extractor = RunnableLambda(self._extract_item_metadata).with_config(
            run_name="ItemMetadataExtractor"
        )
        
        # Step 2: Processing pipeline that runs metadata extraction
        processing_pipeline = RunnableMap({
            "enriched_data": metadata_extractor,
        })
        
        # Step 3: Create prompt template for scoring with JSON output
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert time series analyst specializing in price anomaly detection.

    You will receive price data for an item and must analyze EACH price observation.
    This time series has been chunked into smaller segments for processing.
    Base your analysis on the provided statistics for the entire item and near surrounding prices.

    CRITICAL REQUIREMENTS:
    1. Respond ONLY with valid JSON. No other text before or after.
    2. You MUST provide EXACTLY {n_prices} observations - one for each input price.
    3. Verify: len(observations) == {n_prices} before responding.
    4. Never skip, merge, or omit any price observation.

    ## EXAMPLE 1:
    Input:
    Item: ITEM_001
    Statistics: Mean=$50.00, Std=$5.00, Median=$49.50, Q25=$46.00, Q75=$53.00
    Prices: $48.00, $51.00, $49.50, $85.00, $50.50

    Output:
    {{"observations": [
      {{"label": "NORMAL", "confidence": 0.92, "reason": "Within 0.4 std of mean"}},
      {{"label": "NORMAL", "confidence": 0.88, "reason": "Close to median value"}},
      {{"label": "NORMAL", "confidence": 0.95, "reason": "Matches median exactly"}},
      {{"label": "ANOMALOUS", "confidence": 0.98, "reason": "7 std above mean massive spike"}},
      {{"label": "NORMAL", "confidence": 0.90, "reason": "Within normal range"}}
    ]}}

    ## EXAMPLE 2:
    Input:
    Item: ITEM_002
    Statistics: Mean=$100.00, Std=$10.00, Median=$98.00, Q25=$92.00, Q75=$108.00
    Prices: $95.00, $99.00, $102.00

    Output:
    {{"observations": [
      {{"label": "NORMAL", "confidence": 0.85, "reason": "Within IQR range"}},
      {{"label": "NORMAL", "confidence": 0.93, "reason": "Very close to median"}},
      {{"label": "NORMAL", "confidence": 0.87, "reason": "Within Q75 boundary"}}
    ]}}

    ## EXAMPLE 3:
    Input:
    Item: ITEM_003
    Statistics: Mean=$25.00, Std=$3.00, Median=$24.50, Q25=$22.00, Q75=$27.50
    Prices: $8.00, $24.00, $25.50, $26.00, $15.00, $25.00

    Output:
    {{"observations": [
      {{"label": "ANOMALOUS", "confidence": 0.99, "reason": "5.7 std below mean extreme drop"}},
      {{"label": "NORMAL", "confidence": 0.91, "reason": "Near median expected value"}},
      {{"label": "NORMAL", "confidence": 0.89, "reason": "Within 0.2 std of mean"}},
      {{"label": "NORMAL", "confidence": 0.85, "reason": "Close to Q75 threshold"}},
      {{"label": "ANOMALOUS", "confidence": 0.95, "reason": "3.3 std below mean unusual low"}},
      {{"label": "NORMAL", "confidence": 0.90, "reason": "Matches mean value"}}
    ]}}

    ## DETECTION RULES:
    - ANOMALOUS if: |price - mean| > 2σ OR price outside [Q25 - 1.5*IQR, Q75 + 1.5*IQR]
    - NORMAL otherwise
    - Confidence: Higher for extreme deviations (>3σ = 0.95+), moderate for borderline cases (2-3σ = 0.80-0.90)
    - Reasons: Be specific and concise (max 15 words)

    ## YOUR TASK:
    Item: {item_id}
    Statistics:
    - Mean: ${mean:.2f}
    - Std: ${std:.2f}
    - Median: ${median:.2f}
    - Q25: ${q25:.2f}
    - Q75: ${q75:.2f}

    Prices to analyze: {prices_list}
    Corresponding time indices: {ts_list}
             
    Analyze these {n_prices} prices following the examples above."""),
            ("user", "Return JSON with EXACTLY {n_prices} observations. Count verification required: observations.length MUST equal {n_prices}.")
        ])
        
        def format_for_prompt(data: Dict[str, Any]) -> Dict[str, Any]:
            enriched = data["enriched_data"]
            meta: ModelMetadata = enriched["metadata"]
            prices: np.ndarray = enriched["prices"]
            
            return {
                "item_id": enriched["item_id"],
                "mean": meta.mean,
                "std": meta.std,
                "median": meta.median,
                "q25": meta.q25,
                "q75": meta.q75,
                "prices_list": ", ".join([f"${p:.2f}" for p in prices]),
                "ts_list": ", ".join(enriched["ts_list"]),
                "n_prices": len(prices)
            }
        
        formatter = RunnableLambda(format_for_prompt).with_config(run_name="PromptFormatter")
        
        # parser = StrOutputParser()
        
        chain = (
            processing_pipeline 
            | formatter 
            | prompt_template 
            | self._llm 
            # | parser
        )
        
        return chain
    
    def _parse_llm_response_batch(
        self, 
        response: str, 
        item_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse LLM JSON response for a single item."""
        import json
        
        item_id = item_data["item_id"]
        prices = np.array(item_data["prices"])
        n_prices = len(prices)
        
        labels = []
        confidences = []
        reasons = []
        scores = []
        
        # Parse JSON response
        try:
            response_json = json.loads(response)
            observations = response_json.get("observations", [])
        except json.JSONDecodeError:
            print(f"⚠️ ERROR: Invalid JSON for item {item_id}. Fallback to all NORMAL.")
            observations = []

        # Graceful degradation for length mismatch
        if len(observations) < n_prices:
            missing_count = n_prices - len(observations)
            print(f"⚠️ WARNING: Item {item_id} length mismatch (Exp: {n_prices}, Got: {len(observations)}). Padding {missing_count} entries.")
            for _ in range(missing_count):
                observations.append({
                    "label": "NORMAL",
                    "confidence": 0.0,
                    "reason": "System: Padded due to missing LLM output"
                })
        
        # Process exactly n_prices (truncate if too many, use padded if too few)
        for obs in observations[:n_prices]:
            label = obs.get("label", "NORMAL")
            confidence = float(obs.get("confidence", 0.5))
            reason = obs.get("reason", "No reason provided")
            
            labels.append(label)
            confidences.append(confidence)
            reasons.append(reason)
            
            # Convert to numeric score for compatibility: ANOMALOUS = high, NORMAL = low
            
            
        scores_array = np.array(scores)
    
        return {
            "item_id": item_id,
            "scores": scores_array,
            "labels": labels,
            "confidences": confidences,
            "reasons": reasons,
            "prices": prices
        }
    
    async def _score_all_items_async(self, items_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score all items concurrently using async batching.
        
            LLM Results Example <Dict> Item Id, List[scores], List[Labels], List[Confidences], List[Reasons] </Dict>:
            [
                {'item_id': 'ITEM_001', 'labels': ['ANOMALOUS', 'NORMAL', 'NORMAL', 'ANOMALOUS', 'NORMAL', 'ANOMALOUS', 'NORMAL', 'ANOMALOUS'], 'confidences': [0.98, 0.92, 0.88, 0.95, 0.9, 0.99, 0.85, 0.96], 'reasons': ['7 std below mean massive drop', 'Within normal range', 'Close to median value', '3 std below mean unusual low', 'Within Q75 boundary', '10 std above mean massive spike', 'Near median expected value', '4.5 std below mean extreme drop'], 'prices': array([100., 102.,  98., 101., 600.,  99., 103., 100.])}, 
                {'item_id': 'ITEM_002', 'labels': ['ANOMALOUS', 'NORMAL', 'NORMAL', 'ANOMALOUS', 'NORMAL', 'ANOMALOUS', 'NORMAL', 'ANOMALOUS'], 'confidences': [0.99, 0.91, 0.89, 0.95, 0.9, 0.98, 0.85, 0.99], 'reasons': ['7 std below mean extreme drop', 'Near median expected value', 'Within 0.2 std of mean', '3.3 std below mean unusual low', 'Matches mean value', '4.5 std above mean massive spike', 'Within IQR range', '10 std below mean extreme drop'], 'prices': array([200., 202., 198., 201.,  20., 199., 203., 200.])}, 
                {'item_id': 'ITEM_003', 'labels': ['ANOMALOUS', 'NORMAL', 'NORMAL', 'ANOMALOUS', 'NORMAL', 'ANOMALOUS', 'NORMAL', 'ANOMALOUS'], 'confidences': [0.99, 0.91, 0.89, 0.95, 0.9, 0.98, 0.85, 0.99], 'reasons': ['3.2 std below mean extreme low', 'Near median expected value', 'Within 0.2 std of mean', '1.6 std below mean unusual low', 'Matches mean value', '4.8 std above mean massive spike', 'Within IQR range', '5.6 std below mean extreme drop'], 'prices': array([50., 52., 48., 51., 49., 53., 50., 52.])}, 
                {'item_id': 'ITEM_004', 'labels': ['NORMAL', 'NORMAL', 'NORMAL', 'ANOMALOUS', 'NORMAL', 'ANOMALOUS', 'NORMAL', 'ANOMALOUS'], 'confidences': [0.92, 0.88, 0.95, 0.98, 0.9, 0.99, 0.85, 0.95], 'reasons': ['Within 0.4 std of mean', 'Close to median value', 'Matches median exactly', '7 std above mean massive spike', 'Within normal range', '4.3 std below mean extreme drop', 'Near median expected value', '2.2 std above mean unusual high'], 'prices': array([75., 76., 77., 78., 79., 80., 81., 82.])}, 
                {'item_id': 'ITEM_005', 'labels': ['NORMAL', 'NORMAL', 'ANOMALOUS', 'NORMAL', 'ANOMALOUS', 'NORMAL', 'ANOMALOUS', 'NORMAL'], 'confidences': [0.92, 0.88, 0.98, 0.9, 0.95, 0.85, 0.99, 0.87], 'reasons': ['Within 0.4 std of mean', 'Close to median value', '7 std above mean massive spike', 'Within normal range', '3 std below mean unusual low', 'Near Q25 threshold', '10 std above mean extreme spike', 'Within IQR range'], 'prices': array([150., 152., 800., 151., 149.,  10., 150., 152.])}
            ]
        """
        import time
        
        print(f"\n  Processing {len(items_data)} items concurrently...")
        print(f"  Max concurrent requests: {self.max_concurrent}")
        
        start_time = time.time()
        
        chain = self._create_item_chain()
        results: List[Dict[str, Any]] = []
        n_batches = (len(items_data) - 1) // self.max_concurrent + 1
        
        with mlflow.start_span(name="llm_async_scoring") as scoring_span:
            scoring_span.set_attribute("total_items", len(items_data))
            scoring_span.set_attribute("max_concurrent", self.max_concurrent)
            scoring_span.set_attribute("model_name", self.model_name)
            
            for i in range(0, len(items_data), self.max_concurrent):
                batch = items_data[i:i + self.max_concurrent]
                batch_num = i // self.max_concurrent + 1
                print(f"  Batch {batch_num}/{n_batches}: Processing {len(batch)} items...")
                
                batch_start = time.time()
                
                with mlflow.start_span(name=f"batch_{batch_num}") as batch_span:
                    batch_span.set_attribute("batch_number", batch_num)
                    batch_span.set_attribute("batch_size", len(batch))
                    batch_span.set_attribute("batch_item_ids", ",".join(item["item_id"] for item in batch))
                    
                    with mlflow.start_span(name=f"batch_{batch_num}_llm_call") as call_span:
                        call_span.set_attribute("input_count", len(batch))
                        call_span.set_attribute("temperature", self.temperature)
                        call_span.set_attribute("model_name", self.model_name)
                        batch_results = await chain.abatch(batch)
                        call_span.set_attribute("output_count", len(batch_results))

                    for j, response in enumerate(batch_results):
                        item_data = batch[j]
                        item_id = item_data["item_id"]
                        response_str = response.content if isinstance(response, AIMessage) else str(response)
                        
                        with mlflow.start_span(name=f"item_{item_id}_processing") as item_span:
                            item_span.set_attribute("item_id", item_id)
                            item_span.set_attribute("input_prices", str(item_data["prices"]))
                            item_span.set_attribute("input_timestamps", str(item_data.get("ts_list", [])))
                            item_span.set_attribute("raw_response_preview", response_str[:400])
                            
                            parsed: Dict[str, Any] = self._parse_llm_response_batch(str(response_str), item_data)
                            results.append(parsed)
                            item_span.set_attribute("labels", ",".join(parsed.get("labels", [])))
                            item_span.set_attribute("confidences", ",".join([f"{c:.2f}" for c in parsed.get("confidences", [])]))
                            item_span.set_attribute("reasons_preview", ";".join(parsed.get("reasons", []))[:400])
                
                batch_elapsed = time.time() - batch_start
                self.total_llm_calls += len(batch)
                print(f"    Completed batch {batch_num} in {batch_elapsed:.2f}s")
        
        self.total_latency = time.time() - start_time
        return results
    
    def score(self, items: List[ItemData]) -> None:
        return self.LLMfit_and_detect(items)
    
    @mlflow.trace(name="LLM_Predictor", span_type="CHAIN")
    def LLMfit_and_detect(self, items: List[ItemData]) -> None:
        """
        Score all items concurrently and populate their attributes in-place.
        
        This method mutates the input items by reference, populating:
        - scores: np.ndarray (mapped from confidence)
        - labels: np.ndarray (1 for ANOMALOUS, 0 for NORMAL)
        - reasons: List[str] (explanations for each observation)
        
        Args:
            items: List of ItemData dicts to process. Each must contain:
                - item_id: str
                - prices: np.ndarray
                - timestamps: np.ndarray
                
        Returns:
            None (mutates items in-place)
        """
        
        # Configuration for chunking to avoid token limits
        CHUNK_SIZE = 10  # Safe size: 40 points * ~30 tokens/point = 1200 output tokens
        
        with mlflow.start_span(name="prepare_items") as prep_span:
            items_data: List[Dict[str, Any]] = []
            chunk_map: List[Tuple[int, int, int]] = []  # (original_item_idx, start_idx, end_idx)
            
            for i, item in enumerate(items):
                prices = item["prices"]
                timestamps = item["timestamps"]
                n_points = len(prices)
                
                # Calculate global stats ONCE for the whole item
                # Pass these to every chunk so context remains global
                global_stats = {
                    "mean": float(np.mean(prices)),
                    "std": float(np.std(prices)),
                    "median": float(np.median(prices)),
                    "q25": float(np.percentile(prices, 25)),
                    "q75": float(np.percentile(prices, 75)),
                    "min": float(np.min(prices)),
                    "max": float(np.max(prices)),
                    "n_samples": n_points
                }
                
                # Split into chunks
                for start_idx in range(0, n_points, CHUNK_SIZE):
                    end_idx = min(start_idx + CHUNK_SIZE, n_points)
                    
                    chunk_prices = prices[start_idx:end_idx].tolist()
                    chunk_ts = [str(ts) for ts in timestamps[start_idx:end_idx]]
                    
                    items_data.append({
                        "item_id": f"{item['item_id']}_chunk_{start_idx}",
                        "original_item_id": item["item_id"],
                        "prices": chunk_prices,
                        "ts_list": chunk_ts,
                        "precomputed_stats": global_stats  # Pass global stats to chunk
                    })
                    
                    # Map this chunk back to the original item
                    chunk_map.append((i, start_idx, end_idx))
            
            prep_span.set_attribute("total_original_items", len(items))
            prep_span.set_attribute("total_chunks", len(items_data))
            total_observations: int = sum(len(item["prices"]) for item in items)
            prep_span.set_attribute("total_observations", total_observations)

        with mlflow.start_span(name="llm_scoring") as scoring_span:
            scoring_span.set_attribute("max_concurrent", self.max_concurrent)
            scoring_span.set_attribute("model_name", self.model_name)
            
            # Run async scoring - returns List[Dict] with parsed LLM results
            results: List[Dict[str, Any]] = asyncio.run(self._score_all_items_async(items_data))
            scoring_span.set_attribute("chunks_scored", len(results))

        with mlflow.start_span(name="populate_span") as populate_span:
            # Initialize empty arrays for all items
            for item in items:
                n = len(item["prices"])
                item["scores"] = np.zeros(n, dtype=float)
                item["labels"] = ["NORMAL"] * n  # Keep as list for easier manipulation
                item["reasons"] = [""] * n
            
            # Reassemble chunks into original items by mutating items in-place
            for chunk_idx, result in enumerate(results):
                original_item_idx, start_idx, end_idx = chunk_map[chunk_idx]
                target_item = items[original_item_idx]  # Direct reference to original item
                
                # Get results for this chunk
                chunk_scores = np.array(result["confidences"], dtype=float)
                chunk_labels = result["labels"]  # List of strings: ["NORMAL", "ANOMALOUS", ...]
                chunk_reasons = result["reasons"]  # List of strings
                
                # Verify length match (critical for reassembly)
                expected_len = end_idx - start_idx
                actual_len = len(chunk_scores)
                
                if actual_len != expected_len:
                    # Pad or truncate if LLM messed up length despite instructions
                    if actual_len < expected_len:
                        pad_len = expected_len - actual_len
                        chunk_scores = np.pad(chunk_scores, (0, pad_len), constant_values=0.0)
                        chunk_labels = chunk_labels + ["NORMAL"] * pad_len
                        chunk_reasons = chunk_reasons + ["Error: Missing LLM output"] * pad_len
                    else:
                        chunk_scores = chunk_scores[:expected_len]
                        chunk_labels = chunk_labels[:expected_len]
                        chunk_reasons = chunk_reasons[:expected_len]

                # Assign to correct slice in the original item (mutation by reference)
                target_item["scores"][start_idx:end_idx] = chunk_scores
                
                # Assign labels and reasons element by element (since they're lists)
                for offset, (label, reason) in enumerate(zip(chunk_labels, chunk_reasons)):
                    target_item["labels"][start_idx + offset] = label
                    target_item["reasons"][start_idx + offset] = reason
            
            # Convert labels list to numpy array for consistency
            for item in items:
                item["labels"] = np.array(item["labels"])
            
            total_anomalies: int = sum(
                int(np.sum(item["labels"] == "ANOMALOUS")) for item in items
            )
            populate_span.set_attribute("total_anomalies", int(total_anomalies))
            populate_span.set_attribute("anomaly_rate", float(total_anomalies / total_observations))

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics for LLM calls.
        
        Returns:
            Dict with latency, cost, and throughput metrics
        """
        return {
            "total_latency_seconds": self.total_latency,
            "total_llm_calls": self.total_llm_calls,
            "avg_latency_per_call": self.total_latency / max(self.total_llm_calls, 1),
            "throughput_calls_per_second": self.total_llm_calls / max(self.total_latency, 0.001),
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_concurrent": self.max_concurrent
        }
    
    def _compute_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Not used in concurrent version."""
        raise NotImplementedError("Use score_dataframe_with_details() for concurrent processing.")

class ScoreConverter:
    """Converts continuous anomaly scores to binary labels."""
    
    def __init__(self, percentile: float = 90.0) -> None:
        self.percentile: float = percentile
    
    def convert(self, scores: np.ndarray) -> np.ndarray:
        """Convert scores to binary labels using percentile threshold."""
        threshold: float = np.percentile(scores, self.percentile)
        labels: np.ndarray = (scores >= threshold).astype(int)
        return labels


@dataclass
class ABTestResult:
    pass

class BootstrapEvaluator:
   pass


class AnomalySystem:
    """Orchestrates multiple anomaly detection models with dependency injection."""
    
    def __init__(
        self,
        base_structure: Dict[str, ModelData],
        score_converter: ScoreConverter = ScoreConverter(percentile=90.0),
    ) -> None:
        self.base_structure: Dict[str, ModelData] = base_structure
        self.converter: ScoreConverter = score_converter

 
    
    def fit_and_detect(
        self
    ) -> None:
        """
        Fit models per-item and detect anomalies using ground truth for evaluation.
        It mutates by reference the `self.results` attribute.
        """
        
        required_cols: set = {"ITEM_ID", "ORD_CLOSED_DT", "PRICE"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        model: ModelData
        item: ItemData
        for model in self.base_structure.values():
            model_object: BaseAnomalyModel = model["model_object"]
            if 'LLM' in model["name"]:
                # Mutate each item in-place with LLM concurrent processing
                model_object.score(model['items'])  # LLMfit_and_detect: Process entire DataFrame concurrently
            else:
                for item in model['items']:
                    model_object.fit(X=item["prices"], item_id=item["item_id"]) # Populate metadata with series statistics
                    scores: np.ndarray = model_object.score(X=item["prices"], item_id=item["item_id"])
                    item["scores"] = scores
                    item["labels"] = self.converter.convert(scores)
    
    def print_summary(self) -> None:
        """Print evaluation summary for all models using the base_structure."""
        print("=" * 70)
        print("ANOMALY DETECTION SYSTEM SUMMARY")
        print("=" * 70)
        
        if not self.base_structure:
            print("\nNo results available. Run fit_and_detect() first.")
            return
        
        # Calculate total observations from first model's items
        first_model: ModelData = next(iter(self.base_structure.values()))
        total_observations: int = sum(len(item["prices"]) for item in first_model["items"])
        
        print(f"\nTotal observations: {total_observations:,}")
        print(f"Models evaluated: {len(self.base_structure)}")
        print(f"Items processed: {len(first_model['items'])}")
        
        print("\n" + "-" * 70)
        print(f"{'Model':<25} {'Anomalies':<15} {'Rate':<10} {'Score Range':<20}")
        print("-" * 70)
        
        for model_name, model_data in self.base_structure.items():
            # Aggregate statistics across all items for this model
            all_scores: List[float] = []
            total_anomalies: int = 0
            
            for item in model_data["items"]:
                if len(item["scores"]) > 0:
                    all_scores.extend(item["scores"].tolist())
                    # Count anomalies (label == 1 for numeric, "ANOMALOUS" for string)
                    if isinstance(item["labels"][0], str):
                        total_anomalies += np.sum(np.array(item["labels"]) == "ANOMALOUS")
                    else:
                        total_anomalies += int(np.sum(item["labels"]))
            
            anomaly_rate: float = 100 * total_anomalies / total_observations if total_observations > 0 else 0.0
            score_min: float = float(np.min(all_scores)) if all_scores else 0.0
            score_max: float = float(np.max(all_scores)) if all_scores else 0.0
            
            print(f"{model_name:<25} {total_anomalies:<15,} {anomaly_rate:<10.2f}% [{score_min:.2f}, {score_max:.2f}]")
        
        print("-" * 70)
        
        # Optional: Print sample anomalies per model
        print("\n" + "=" * 70)
        print("SAMPLE ANOMALIES PER MODEL (First 3 per item)")
        print("=" * 70)
        
        for model_name, model_data in self.base_structure.items():
            print(f"\n{model_name}:")
            
            for item in model_data["items"][:3]:  # Show first 3 items
                item_id: str = item["item_id"]
                
                # Find anomalous indices
                if isinstance(item["labels"][0], str):
                    anomaly_mask: np.ndarray = np.array(item["labels"]) == "ANOMALOUS"
                else:
                    anomaly_mask = item["labels"] == 1
                
                anomaly_indices: np.ndarray = np.where(anomaly_mask)[0]
                
                if len(anomaly_indices) > 0:
                    print(f"  Item {item_id}: {len(anomaly_indices)} anomalies detected")
                    
                    for idx in anomaly_indices:  # Show first 3 anomalies per item
                        price: float = float(item["prices"][idx])
                        score: float = float(item["scores"][idx])
                        timestamp: str = str(item["timestamps"][idx])
                        reason: str = item["reasons"][idx] if item["reasons"] else "N/A"
                        
                        print(f"    [{timestamp}] Price=${price:.2f}, Score={score:.2f}, Reason: {reason}")
        
def load_real_data(file_path: str) -> pd.DataFrame:
    """Load real price history data."""
    return pd.read_csv(file_path)

if __name__ == "__main__":
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Validate OpenAI API key is present
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please add it to your .env file.")
    
    print("=" * 70)
    print("OFFLINE ANOMALY DETECTION - PER-ITEM RETROSPECTIVE ANALYSIS")
    print("=" * 70)
    
    print("\nLoading historical price data...")
    parser = argparse.ArgumentParser(description='Offline anomaly detection on historical price data')
    parser.add_argument(
        '--absolute_path_to_csv',
        type=str,
        required=True,
        help='Absolute path to the CSV file containing historical price data'
    )
    args = parser.parse_args()
    
    df: pd.DataFrame = load_real_data(args.absolute_path_to_csv)
    # df = df[df['ITEM_ID'].isin(df['ITEM_ID'].unique()[:3])]
    df = df[df['ITEM_ID'].isin(['MLB4097203966', 'MLB4184697886', 'MLB4281133192'])]
    
    required_cols = {'ITEM_ID', 'ORD_CLOSED_DT', 'PRICE'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    n_items = df['ITEM_ID'].nunique()
    n_points = len(df)
    print(f"\nDataset: {n_points} price points across {n_items} items")
    print(f"Overall statistics: mean={df['PRICE'].mean():.2f}, std={df['PRICE'].std():.2f}")
    print(f"Price range: [{df['PRICE'].min():.2f}, {df['PRICE'].max():.2f}]")
    
    print("\n" + "=" * 70)
    print("INITIALIZING COMPONENTS (Dependency Injection)")
    print("=" * 70)
    
    converter = ScoreConverter(percentile=90.0)
    
    zscore_model = ZScoreModel(threshold=3.0)
    quantile_model = MovingQuantileModel(window_size=10, quantile=0.95)
    llm_model = NewLLMAnomalyDetector(
        base_model=ChatOpenAI(
            temperature=0.0,
            model='gpt-4o-mini-2024-07-18',
            response_format={"type": "json_object"},
            api_key=openai_api_key,
            max_tokens=10000
        ),
        window_size=10,
        model_name='gpt-3.5-turbo-0125',
        temperature=0.0,
        max_concurrent=5
    )

    models: List[Tuple[str, BaseAnomalyModel]] = [
        ("ZScore", zscore_model),
        # ("MovingQuantile", quantile_model),
        # ("MAD (robust)", mad_model),
        # ("LocalOutlier", local_outlier_model),
        ("LLM", llm_model),
    ]

    # dict:
    #     model: -> ModelData
    #     {
    #         model_object: BaseAnomalyModel
    #         name: str
    #         items: List[ItemData] where each dict contains:
    #             item_id: str
    #             prices: np.ndarray
    #             timestamps: np.ndarray
    #             scores: np.ndarray
    #             labels: np.ndarray
    #             reasons: Optional[List[str]]
    #    }
    
    
    detailed_results_dict: Dict[str, ModelData] = {}

    for model_name, model_object in models:
        items_list: List[ItemData] = []
        
        for item_id in df['ITEM_ID'].unique():
            item_mask: pd.Series = df['ITEM_ID'] == item_id
            
            item_data: ItemData = {
                "item_id": str(item_id),
                "prices": df.loc[item_mask, 'PRICE'].values.astype(float),
                "timestamps": df.loc[item_mask, 'ORD_CLOSED_DT'].values,
                "scores": np.array([]),
                "labels": np.array([]),
                "reasons": None
            }
            items_list.append(item_data)
        
        detailed_results_dict[model_name] = {
            "model_object": model_object,
            "name": model_name,
            "items": items_list
        }
    
    system = AnomalySystem(
        base_structure=detailed_results_dict,
        score_converter=converter
    )

    print("\n" + "=" * 70)
    print("INITIALIZATION COMPLETE")
    print("=" * 70) 
    print("\nFitting models per item and detecting anomalies...")
    print("(Retrospective analysis - each item processed independently)")

    system.fit_and_detect() # Mutates system.results
    
    print("\n")
    system.print_summary()
    
    print("\n" + "=" * 70)
    print("DETAILED ANOMALY DETECTION RESULTS")
    print("=" * 70)
    
    
    print("\n" + "=" * 70)
    print("FINAL")
    print("=" * 70)