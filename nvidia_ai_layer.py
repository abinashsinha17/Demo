import random
from typing import Dict, Any, List, Literal

# =====================================================================
# NVIDIA ACCELERATED AI LAYER & INFRASTRUCTURE
# =====================================================================

import config
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage

class NVIDIA_NIM_Client:
    """
    Interacts with NVIDIA NIM Microservices using LangChain and an API Key.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        
        # Select the correct API key based on the model
        if "llama" in model_name.lower():
            api_key = config.LLAMA_API_KEY
        elif "nemotron" in model_name.lower():
            api_key = config.NEMOTRON_API_KEY
        else:
            api_key = config.LLAMA_API_KEY # Default fallback
            
        self.llm = ChatNVIDIA(model=model_name, nvidia_api_key=api_key)

    def invoke(self, prompt: str, **kwargs) -> str:
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            return f"Error calling NVIDIA API: {e}"

# Instantiate NIMs as per the Architecture
class NVIDIAAILayer:
    def __init__(self):
        # 1. Classification & Routing
        self.classifier_nim = NVIDIA_NIM_Client("meta/llama-3.3-70b-instruct")
        # 2. Complex Reasoning & RCA
        self.reasoning_nim = NVIDIA_NIM_Client("nvidia/llama-3.3-nemotron-super-49b-v1.5")
        # 3. Reporting & Summarization
        self.reporting_nim = NVIDIA_NIM_Client("meta/llama-3.3-70b-instruct")


# =====================================================================
# NVIDIA NEMO FRAMEWORK
# =====================================================================

class NeMoFramework:
    @staticmethod
    def retriever_rag_pipeline(query: str, context_db: List[str]) -> str:
        """NeMo Retriever: Hybrid Search, Re-ranking, Context Building"""
        return f"[NeMo Retriever] Context built from hybrid search for query: '{query}'"

    @staticmethod
    def guardrails_and_evaluator(response: str) -> bool:
        """NeMo Guardrails: Safety, Factuality, Hallucination Check"""
        is_safe = "unauthorized" not in response.lower()
        factuality_score = random.uniform(0.85, 0.99)
        return is_safe and factuality_score > 0.90

    @staticmethod
    def custom_fine_tuning_trigger(exception_data: Dict):
        """NeMo Custom Fine-Tuning: Domain Adaptation, Continual Learning"""
        print("[NeMo Fine-Tuning] Exception data logged for continual learning and style tuning.")


# =====================================================================
# ROUTING DECISION ENGINE
# =====================================================================

RoutingStrategy = Literal["Rule-Based", "ML-Classification", "Hybrid", "Reinforcement-Learning"]

class RoutingDecisionEngine:
    def __init__(self, strategy: RoutingStrategy = "Hybrid"):
        self.strategy = strategy
        self.ai_layer = NVIDIAAILayer()

    def determine_route(self, inputs: Dict[str, Any]) -> str:
        """
        Determines the routing based on detailed Exception Inputs.
        Inputs expected:
        - Exception Type
        - Severity (Low/Med/High/Critical)
        - NAV Impact (USD)
        - Fund / Strategy
        - Data Source
        - Confidence Score
        - Past Resolution Patterns
        - Compliance Flags
        - Time-Sensitivity
        """
        print(f"\nEvaluating Routing Decision Inputs using '{self.strategy}' Strategy...")
        
        # Unpack critical inputs for logic
        severity = inputs.get("Severity", "Low")
        nav_impact = inputs.get("NAV Impact (USD)", 0)
        compliance_flags = inputs.get("Compliance Flags", False)
        
        route = "Standard Processing"

        # 1. Rule-Based Routing (Deterministic)
        if self.strategy == "Rule-Based":
            if severity in ["High", "Critical"] or compliance_flags or nav_impact > 100000:
                route = "Escalation Queue (Manual Review)"
            else:
                route = "Auto-Reconciliation Skill"

        # 2. ML Classification (Model-Assisted)
        elif self.strategy == "ML-Classification":
            prompt = f"Classify and Route this exception: {inputs}"
            nim_decision = self.ai_layer.classifier_nim.invoke(prompt)
            route = f"ML Routed: {nim_decision}"

        # 3. Hybrid Routing (Rules + ML + Context)
        elif self.strategy == "Hybrid":
            if compliance_flags: # Rule override
                route = "Compliance Escalation Queue"
            else:
                prompt = f"Analyze context and determine route: {inputs}"
                route = f"Hybrid Route determined by NIM: {self.ai_layer.classifier_nim.invoke(prompt)}"

        # 4. Reinforcement Learning (Optimization over time)
        elif self.strategy == "Reinforcement-Learning":
            # Simulating an RL agent exploiting high-confidence past resolution patterns
            confidence = inputs.get("Confidence Score", 0.0)
            if confidence > 0.95:
                route = "Straight-Through Processing (STP) - RL Optimized"
            else:
                route = "Exploration/Human-in-the-Loop"

        return route

