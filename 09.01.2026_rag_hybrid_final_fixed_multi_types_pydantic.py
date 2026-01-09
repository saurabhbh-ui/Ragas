"""
RAG Evaluation - Hybrid Test Set Generation (RAGAS + Custom)
With Pydantic Structured Outputs using LangChain

RAGAS Version: 0.1.21 (Old API)
Reference: https://docs.ragas.io/en/v0.1.21/getstarted/testset_generation.html
LangChain Structured Output: https://docs.langchain.com/oss/python/langchain/structured-output
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
from tqdm import tqdm
from pydantic import BaseModel, Field

# LangChain
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document

# RAGAS (v0.1.21)
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

# Visualization
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# =====================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# =====================================================================

class QuestionClassification(BaseModel):
    """Structured classification for questions"""
    question_style: Literal[
        "chatbot_style", "direct_factual", "procedural", "scenario",
        "analytical", "compliance", "descriptive", "multi_hop",
        "comparative", "conditional"
    ] = Field(description="The style/category of the question")
    
    question_type: Literal[
        "chatbot_style", "direct_factual", "procedural", "scenario",
        "analytical", "compliance", "descriptive", "multi_hop",
        "comparative", "conditional"
    ] = Field(description="The type of the question")
    
    complexity_level: Literal["easy", "medium", "hard"] = Field(
        description="Complexity: easy (single-hop), medium (2-3 hops), hard (multi-hop synthesis)"
    )
    
    reasoning_requirement: Literal["simple", "contextual", "reasoning", "synthesis"] = Field(
        description="Reasoning type: simple (fact lookup), contextual (understand context), reasoning (logical inference), synthesis (combine multiple pieces)"
    )


class CustomQuestion(BaseModel):
    """Structured output for custom generated questions"""
    question: str = Field(description="The generated question")
    answer: str = Field(description="Complete answer based on the context")
    question_type: str = Field(description="Type of question generated")
    complexity: Literal["easy", "medium", "hard"] = Field(
        default="medium",
        description="Complexity level of the question"
    )


class FollowupQuestion(BaseModel):
    """Structured output for follow-up questions"""
    followup_question: str = Field(description="The follow-up question")
    followup_answer: str = Field(description="Answer to the follow-up question")


# =====================================================================
# CONFIGURATION
# =====================================================================

@dataclass
class HybridConfig:
    """Hybrid Generation Configuration"""
    
    # Azure OpenAI
    azure_endpoint: str = "https://your-endpoint.openai.azure.com/"
    azure_api_key: str = "your-api-key"
    azure_api_version: str = "2024-02-01"
    azure_deployment_gpt4: str = "gpt-4"
    azure_deployment_embedding: str = "text-embedding-ada-002"
    
    # Coverage
    num_spatial_regions: int = 4
    questions_per_document: int = 50  # Total questions (RAGAS + Custom combined)
    
    # Chunking
    chunk_size: int = 1500
    chunk_overlap: int = 300
    
    # Generation Split
    ragas_percentage: float = 0.50  # 50% RAGAS (25 questions if total=50)
    custom_percentage: float = 0.50  # 50% Custom (25 questions if total=50)
    
    # Custom Question Types
    custom_question_types: Dict[str, float] = field(default_factory=lambda: {
        "chatbot_style": 0.40,
        "scenario": 0.20,
        "multi_hop": 0.20,
        "comparative": 0.10,
        "conditional": 0.10,
    })
    
    # Multi-turn
    custom_multiturn_ratio: float = 0.50
    max_turns_per_conversation: int = 2
    
    # Domain
    domain_name: str = "Banking and Financial Services (BIS)"
    domain_context: str = "BIS Meeting Services, compliance, operations"
    
    # Quality
    min_quality_score: float = 7.0
    
    # LLM
    temperature: float = 0.7
    max_tokens: int = 3000


def main():
    """Main execution function"""
    
    # Initialize config
    config = HybridConfig()
    
    # UPDATE YOUR CREDENTIALS HERE
    config.azure_endpoint = "https://your-endpoint.openai.azure.com/"
    config.azure_api_key = "your-api-key-here"
    config.azure_deployment_gpt4 = "gpt-4"
    config.azure_deployment_embedding = "text-embedding-ada-002"
    
    # Calculate actual numbers
    ragas_count = int(config.questions_per_document * config.ragas_percentage)
    custom_count = int(config.questions_per_document * config.custom_percentage)
    questions_per_region = config.questions_per_document // config.num_spatial_regions
    
    print("✅ Configuration loaded")
    print(f"   Total questions per doc: {config.questions_per_document}")
    print(f"   RAGAS questions: {ragas_count} ({config.ragas_percentage*100:.0f}%)")
    print(f"   Custom questions: {custom_count} ({config.custom_percentage*100:.0f}%)")
    print(f"   Questions per region: {questions_per_region}")
    
    # =====================================================================
    # INITIALIZE LLMs WITH STRUCTURED OUTPUT
    # =====================================================================
    
    print("\n📍 Initializing LLMs...")
    
    # Base LLM
    base_llm = AzureChatOpenAI(
        azure_endpoint=config.azure_endpoint,
        api_key=config.azure_api_key,
        api_version=config.azure_api_version,
        deployment_name=config.azure_deployment_gpt4,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    
    # Structured LLMs
    classification_llm = base_llm.with_structured_output(QuestionClassification)
    custom_question_llm = base_llm.with_structured_output(CustomQuestion)
    followup_llm = base_llm.with_structured_output(FollowupQuestion)
    
    # Embeddings
    embeddings_model = AzureOpenAIEmbeddings(
        azure_endpoint=config.azure_endpoint,
        api_key=config.azure_api_key,
        api_version=config.azure_api_version,
        deployment=config.azure_deployment_embedding,
    )
    
    print("✅ LLMs initialized with structured output")
    
    # =====================================================================
    # LOAD & CHUNK DOCUMENTS
    # =====================================================================
    
    print("\n📍 Loading documents...")
    
    DOCUMENTS_DIR = "./documents"
    Path(DOCUMENTS_DIR).mkdir(parents=True, exist_ok=True)
    
    documents = load_documents(DOCUMENTS_DIR)
    print(f"✅ Loaded {len(documents)} pages/sections")
    
    if not documents:
        print("⚠️  No documents found. Add files to ./documents/")
        print("   Supported formats: PDF, DOCX, MD, TXT")
        return
    
    print("\n📍 Chunking documents...")
    chunked_docs_by_file = chunk_documents(documents, config)
    print("✅ Chunking complete")
    
    # =====================================================================
    # SAMPLE CHUNKS
    # =====================================================================
    
    print("\n📍 Sampling chunks...")
    sampled_for_ragas, sampled_for_custom = sample_chunks_for_both(
        chunked_docs_by_file, config, ragas_count, custom_count
    )
    print("✅ Sampling complete")
    
    # =====================================================================
    # RAGAS GENERATION
    # =====================================================================
    
    print(f"\n📍 RAGAS Generation ({ragas_count} questions per doc)...")
    
    ragas_df = generate_ragas_questions(
        base_llm, embeddings_model, sampled_for_ragas, ragas_count
    )
    
    print(f"✅ RAGAS generated {len(ragas_df)} questions")
    
    # =====================================================================
    # CUSTOM GENERATION WITH STRUCTURED OUTPUT
    # =====================================================================
    
    print(f"\n📍 Custom Generation ({custom_count} questions per doc)...")
    
    custom_questions_all = generate_custom_questions_structured(
        sampled_for_custom,
        custom_question_llm,
        followup_llm,
        config,
        custom_count
    )
    
    print(f"✅ Custom generated {len(custom_questions_all)} questions")
    
    # =====================================================================
    # CLASSIFY ALL QUESTIONS WITH STRUCTURED OUTPUT
    # =====================================================================
    
    print("\n📍 Classifying questions with structured output...")
    
    # Classify RAGAS questions
    if len(ragas_df) > 0:
        ragas_df = classify_ragas_questions_structured(
            ragas_df, classification_llm
        )
        print(f"✅ Classified {len(ragas_df)} RAGAS questions")
    
    # Re-classify Custom questions
    RECLASSIFY_CUSTOM = True
    
    if RECLASSIFY_CUSTOM and len(custom_questions_all) > 0:
        custom_questions_all = reclassify_custom_questions_structured(
            custom_questions_all, classification_llm
        )
        print(f"✅ Re-classified {len(custom_questions_all)} custom questions")
    
    # =====================================================================
    # MERGE & CREATE FINAL DATASET
    # =====================================================================
    
    print("\n📍 Merging datasets...")
    
    final_df = merge_datasets(ragas_df, custom_questions_all)
    
    # ENSURE EXACT COUNT
    if len(final_df) > config.questions_per_document:
        print(f"⚠️  Trimming from {len(final_df)} to {config.questions_per_document} questions")
        final_df = final_df.head(config.questions_per_document)
    
    if len(final_df) > 0:
        print(f"✅ Final dataset: {len(final_df)} questions")
        print(f"   RAGAS: {len(final_df[final_df['generation_method'] == 'ragas'])}")
        print(f"   Custom: {len(final_df[final_df['generation_method'] == 'custom'])}")
        
        # Coverage analysis
        print_coverage_analysis(final_df)
        
        # Sample questions
        print_sample_questions(final_df)
        
        # Export
        export_datasets(final_df)
        
        # Visualize
        create_visualizations(final_df)
        
        print("\n✅ Complete! Check ./outputs/ for generated files")
    else:
        print("⚠️  No questions generated")


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def load_documents(documents_dir: str) -> List[Document]:
    """Load documents from multiple file types: PDF, DOCX, MD, TXT"""
    documents = []
    doc_dir = Path(documents_dir)
    
    print("\nLoading documents...")
    
    # Handle PDF files
    pdf_files = list(doc_dir.glob("*.pdf"))
    if pdf_files:
        print(f"\n  📄 Processing PDF files:")
        for file_path in tqdm(pdf_files, desc="  Loading PDFs"):
            try:
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
                
                for doc in docs:
                    doc.metadata["source_file"] = file_path.name
                    doc.metadata["file_type"] = "pdf"
                
                documents.extend(docs)
                print(f"    ✓ {file_path.name}: {len(docs)} pages")
                
            except Exception as e:
                print(f"    ✗ Error loading {file_path.name}: {e}")
    
    # Handle DOCX files
    docx_files = list(doc_dir.glob("*.docx"))
    if docx_files:
        print(f"\n  📄 Processing DOCX files:")
        for file_path in tqdm(docx_files, desc="  Loading DOCX"):
            try:
                loader = Docx2txtLoader(str(file_path))
                docs = loader.load()
                
                for doc in docs:
                    doc.metadata["source_file"] = file_path.name
                    doc.metadata["file_type"] = "docx"
                
                documents.extend(docs)
                print(f"    ✓ {file_path.name}: {len(docs)} pages")
                
            except Exception as e:
                print(f"    ✗ Error loading {file_path.name}: {e}")
    
    # Handle text-based files (MD, TXT) with simple read_text
    text_extensions = ['.md', '.txt']
    for ext in text_extensions:
        text_files = list(doc_dir.glob(f"*{ext}"))
        
        if text_files:
            print(f"\n  📄 Processing {ext.upper()} files:")
            for file_path in tqdm(text_files, desc=f"  Loading {ext}"):
                try:
                    # Simple UTF-8 text reading
                    text = file_path.read_text(encoding='utf-8')
                    
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source_file": file_path.name,
                            "file_type": ext.replace('.', '')
                        }
                    )
                    
                    documents.append(doc)
                    print(f"    ✓ {file_path.name}: 1 document")
                    
                except Exception as e:
                    print(f"    ✗ Error loading {file_path.name}: {e}")
    
    return documents


def chunk_documents(documents: List[Document], config: HybridConfig) -> Dict[str, List[Document]]:
    """Chunk documents by source file"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    docs_by_file = {}
    for doc in documents:
        source = doc.metadata.get("source_file", "unknown")
        if source not in docs_by_file:
            docs_by_file[source] = []
        docs_by_file[source].append(doc)
    
    chunked_docs = {}
    for source, docs in docs_by_file.items():
        combined_text = "\n\n".join([d.page_content for d in docs])
        combined_doc = Document(page_content=combined_text, metadata={"source_file": source})
        chunks = text_splitter.split_documents([combined_doc])
        
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
        
        chunked_docs[source] = chunks
        print(f"  ✓ {source}: {len(chunks)} chunks")
    
    return chunked_docs


def sample_chunks_spatial(chunks: List[Document], n_samples: int, num_regions: int) -> List[Document]:
    """Sample chunks evenly from spatial regions"""
    total = len(chunks)
    if total == 0:
        return []
    
    region_size = total // num_regions
    samples_per_region = n_samples // num_regions
    
    sampled = []
    for region_idx in range(num_regions):
        start = region_idx * region_size
        end = start + region_size if region_idx < num_regions - 1 else total
        region_chunks = chunks[start:end]
        
        if len(region_chunks) > 0:
            step = max(1, len(region_chunks) // samples_per_region)
            selected = region_chunks[::step][:samples_per_region]
            for chunk in selected:
                chunk.metadata["region_id"] = region_idx
            sampled.extend(selected)
    
    return sampled[:n_samples]


def sample_chunks_for_both(chunked_docs: Dict, config: HybridConfig, ragas_count: int, custom_count: int):
    """Sample chunks for both RAGAS and Custom"""
    sampled_for_ragas = {}
    sampled_for_custom = {}
    
    for source, chunks in chunked_docs.items():
        # Sample for RAGAS
        sampled_for_ragas[source] = sample_chunks_spatial(chunks, ragas_count, config.num_spatial_regions)
        
        # Sample for Custom (different chunks)
        all_indices = set(range(len(chunks)))
        ragas_indices = set([c.metadata['chunk_index'] for c in sampled_for_ragas[source]])
        remaining_indices = list(all_indices - ragas_indices)
        
        if len(remaining_indices) >= custom_count:
            step = max(1, len(remaining_indices) // custom_count)
            selected_indices = remaining_indices[::step][:custom_count]
            sampled_for_custom[source] = [chunks[i] for i in selected_indices]
        else:
            sampled_for_custom[source] = sample_chunks_spatial(chunks, custom_count, config.num_spatial_regions)
        
        print(f"{source}: RAGAS={len(sampled_for_ragas[source])}, Custom={len(sampled_for_custom[source])}")
    
    return sampled_for_ragas, sampled_for_custom


def generate_ragas_questions(llm, embeddings, sampled_chunks, target_count):
    """Generate questions using RAGAS v0.1.21 - handles any document size"""
    all_questions = []
    
    # Initialize generator (v0.1.21 API)
    generator = TestsetGenerator.from_langchain(
        generator_llm=llm,
        critic_llm=llm,
        embeddings=embeddings
    )
    
    for source_file, chunks in sampled_chunks.items():
        print(f"\n{'='*70}")
        print(f"RAGAS: {source_file}")
        print(f"{'='*70}")
        
        n_chunks = len(chunks)
        print(f"Chunks available: {n_chunks}")
        print(f"Generating {target_count} questions...")
        
        if n_chunks < target_count:
            print(f"  ⚠️  Only {n_chunks} chunks, RAGAS will generate multiple questions per chunk")
        
        try:
            # RAGAS handles multiple questions per chunk automatically
            testset = generator.generate_with_langchain_docs(
                chunks, 
                test_size=target_count,  # Use target directly
                distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25}
            )
            
            # Convert to pandas
            df = testset.to_pandas()
            df["source_file"] = source_file
            df["generation_method"] = "ragas"
            all_questions.append(df)
            print(f"✅ Generated {len(df)} questions")
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    if all_questions:
        result = pd.concat(all_questions, ignore_index=True)
        # Ensure exact count
        if len(result) > target_count:
            result = result.head(target_count)
        return result
    return pd.DataFrame()


def generate_custom_questions_structured(sampled_chunks, question_llm, followup_llm, config, target_count):
    """Generate custom questions using structured output - handles any document size"""
    all_questions = []
    
    for source_file, chunks in sampled_chunks.items():
        print(f"\n{'='*70}")
        print(f"Custom: {source_file}")
        print(f"{'='*70}")
        
        n_chunks = len(chunks)
        
        # HANDLE SMALL DOCUMENTS: Cycle through chunks if not enough
        if n_chunks < target_count:
            print(f"  ⚠️  Only {n_chunks} chunks, need {target_count} questions")
            print(f"  Cycling through chunks...")
            
            # Repeat chunks to match target
            chunks_to_use = []
            for i in range(target_count):
                chunk_idx = i % n_chunks  # Cycle: 0,1,2,0,1,2,0,1,2...
                chunks_to_use.append(chunks[chunk_idx])
            
            print(f"  Using {len(chunks_to_use)} chunk positions (from {n_chunks} unique chunks)")
        else:
            # Enough chunks
            chunks_to_use = chunks[:target_count]
        
        # Get question type sequence
        q_types = get_type_sequence(len(chunks_to_use), config.custom_question_types)
        
        # Determine multi-turn
        n_multiturn = int(len(q_types) * config.custom_multiturn_ratio)
        multiturn_indices = np.random.choice(len(q_types), n_multiturn, replace=False)
        
        doc_questions = []
        
        for idx, (chunk, q_type) in enumerate(tqdm(list(zip(chunks_to_use, q_types)), desc="Custom Gen")):
            # Generate question with structured output
            qa = generate_custom_question_with_structure(chunk, q_type, question_llm, config)
            
            if qa:
                qa['conversation_type'] = 'multi_turn' if idx in multiturn_indices else 'single_turn'
                qa['generation_method'] = 'custom'
                qa['has_followup'] = False
                qa['followup_questions'] = []
                
                # Generate follow-up if multi-turn
                if idx in multiturn_indices:
                    followup = generate_followup_with_structure(qa, chunk, followup_llm)
                    if followup:
                        qa['has_followup'] = True
                        qa['followup_questions'] = [followup]
                
                doc_questions.append(qa)
        
        all_questions.extend(doc_questions)
        print(f"✅ Generated {len(doc_questions)} questions")
        print(f"   Multi-turn: {sum(1 for q in doc_questions if q['has_followup'])}")
    
    return all_questions


def generate_custom_question_with_structure(chunk: Document, q_type: str, llm, config) -> Optional[Dict]:
    """Generate custom question using structured output"""
    prompt = f"""You are generating test questions for {config.domain_name}.

CONTEXT:
{chunk.page_content[:2500]}

QUESTION TYPE: {q_type}

TYPE DEFINITIONS:
- chatbot_style: Conversational ("Can you help me understand...?")
- scenario: Realistic situation ("A client requests... What should...?")
- multi_hop: Requires multiple pieces ("If X and Y, then what...?")
- comparative: Compare items ("What's the difference between...?")
- conditional: If-then ("If this happens, what are the consequences?")

REQUIREMENTS:
- Be SPECIFIC and PROFESSIONAL
- Include domain context
- Make it ANSWERABLE from the context

Generate ONE {q_type} question with its complete answer.
"""
    
    try:
        result: CustomQuestion = llm.invoke(prompt)
        return {
            'question': result.question,
            'answer': result.answer,
            'question_type': result.question_type,
            'complexity': result.complexity,
            'source_file': chunk.metadata.get('source_file'),
            'region_id': chunk.metadata.get('region_id', 0)
        }
    except Exception as e:
        print(f"  ⚠️  Error: {str(e)[:80]}")
        return None


def generate_followup_with_structure(initial_qa: Dict, chunk: Document, llm) -> Optional[Dict]:
    """Generate follow-up question using structured output"""
    prompt = f"""Generate a FOLLOW-UP question.

CONTEXT: {chunk.page_content[:2500]}
INITIAL Q: {initial_qa['question']}
INITIAL A: {initial_qa['answer']}

Generate a natural follow-up that builds on the initial question.
"""
    
    try:
        result: FollowupQuestion = llm.invoke(prompt)
        return {
            'followup_question': result.followup_question,
            'followup_answer': result.followup_answer
        }
    except Exception as e:
        return None


def classify_ragas_questions_structured(ragas_df: pd.DataFrame, llm) -> pd.DataFrame:
    """Classify RAGAS questions using structured output"""
    print(f"\nClassifying {len(ragas_df)} RAGAS questions...")
    
    classifications = []
    for idx, row in tqdm(ragas_df.iterrows(), total=len(ragas_df), desc="Classifying"):
        classification = classify_question_with_structure(
            row['question'],
            row.get('answer', row.get('ground_truth', '')),
            llm
        )
        classifications.append(classification)
    
    class_df = pd.DataFrame(classifications)
    ragas_df = pd.concat([ragas_df, class_df], axis=1)
    
    # Add metadata
    ragas_df['conversation_type'] = 'single_turn'
    ragas_df['is_followup'] = False
    ragas_df['turn_number'] = 1
    ragas_df['parent_question_id'] = None
    
    return ragas_df


def classify_question_with_structure(question: str, answer: str, llm) -> Dict:
    """Classify a question using structured output"""
    prompt = f"""You are a question classifier for RAG evaluation test sets.

QUESTION: {question}
ANSWER: {answer[:500]}

Analyze this question and classify it according to the following categories:

QUESTION STYLE (choose ONE):
- chatbot_style: Conversational, help-seeking
- direct_factual: Direct fact questions
- procedural: Process/how-to questions
- scenario: Realistic situation-based
- analytical: Analysis/evaluation
- compliance: Regulatory/policy
- descriptive: Detailed description
- multi_hop: Requires multiple pieces of info
- comparative: Comparison questions
- conditional: If-then scenarios

COMPLEXITY LEVEL:
- easy: Single-hop, direct retrieval
- medium: 2-3 hops, some reasoning
- hard: Multi-hop, synthesis required

REASONING REQUIREMENT:
- simple: Direct fact lookup
- contextual: Requires understanding context
- reasoning: Requires logical reasoning
- synthesis: Requires combining multiple pieces

Classify this question.
"""
    
    try:
        result: QuestionClassification = llm.invoke(prompt)
        return {
            'question_style': result.question_style,
            'question_type': result.question_type,
            'complexity_level': result.complexity_level,
            'reasoning_requirement': result.reasoning_requirement
        }
    except Exception as e:
        print(f"  ⚠️  Error: {str(e)[:60]}")
        return {
            'question_style': 'direct_factual',
            'question_type': 'direct_factual',
            'complexity_level': 'medium',
            'reasoning_requirement': 'contextual'
        }


def reclassify_custom_questions_structured(custom_questions: List[Dict], llm) -> List[Dict]:
    """Re-classify custom questions using structured output"""
    print(f"\nRe-classifying {len(custom_questions)} custom questions...")
    
    for qa in tqdm(custom_questions, desc="Re-classifying"):
        classification = classify_question_with_structure(qa['question'], qa['answer'], llm)
        qa.update(classification)
    
    return custom_questions


def get_type_sequence(n: int, distribution: Dict[str, float]) -> List[str]:
    """Create question type sequence"""
    seq = []
    for q_type, ratio in distribution.items():
        count = max(1, int(n * ratio))
        seq.extend([q_type] * count)
    import random
    random.shuffle(seq)
    return seq[:n]


def merge_datasets(ragas_df: pd.DataFrame, custom_questions: List[Dict]) -> pd.DataFrame:
    """Merge RAGAS and Custom datasets"""
    
    # Flatten custom questions
    custom_flat = []
    for qa in custom_questions:
        initial = {
            'question': qa['question'],
            'answer': qa['answer'],
            'source_file': qa.get('source_file'),
            'region_id': qa.get('region_id', 0) + 1,
            'question_type': qa.get('question_type'),
            'question_style': qa.get('question_style', qa.get('question_type')),
            'complexity_level': qa.get('complexity_level', qa.get('complexity', 'medium')),
            'reasoning_requirement': qa.get('reasoning_requirement', 'contextual'),
            'conversation_type': qa.get('conversation_type'),
            'generation_method': 'custom',
            'is_followup': False,
            'turn_number': 1,
            'parent_question_id': None,
        }
        custom_flat.append(initial)
        
        # Follow-ups
        if qa.get('has_followup'):
            parent_idx = len(custom_flat) - 1
            for followup in qa.get('followup_questions', []):
                fu = {
                    'question': followup['followup_question'],
                    'answer': followup['followup_answer'],
                    'source_file': qa.get('source_file'),
                    'region_id': qa.get('region_id', 0) + 1,
                    'question_type': 'followup',
                    'question_style': 'followup',
                    'complexity_level': qa.get('complexity_level', 'medium'),
                    'reasoning_requirement': qa.get('reasoning_requirement', 'contextual'),
                    'conversation_type': 'multi_turn',
                    'generation_method': 'custom',
                    'is_followup': True,
                    'turn_number': 2,
                    'parent_question_id': parent_idx,
                }
                custom_flat.append(fu)
    
    custom_df = pd.DataFrame(custom_flat)
    
    # Standardize RAGAS
    if len(ragas_df) > 0:
        ragas_standardized = ragas_df.rename(columns={'ground_truth': 'answer'})
        if 'region_id' not in ragas_standardized.columns:
            ragas_standardized['region_id'] = 0
    else:
        ragas_standardized = pd.DataFrame()
    
    # Combine
    all_dfs = []
    if len(ragas_standardized) > 0:
        all_dfs.append(ragas_standardized)
    if len(custom_df) > 0:
        all_dfs.append(custom_df)
    
    if all_dfs:
        common_cols = ['question', 'answer', 'source_file', 'generation_method',
                       'question_type', 'question_style', 'complexity_level', 'reasoning_requirement',
                       'conversation_type', 'is_followup', 'turn_number', 'parent_question_id']
        
        for df in all_dfs:
            for col in common_cols:
                if col not in df.columns:
                    df[col] = 'contextual' if col == 'reasoning_requirement' else None
        
        final_df = pd.concat([df[common_cols] for df in all_dfs], ignore_index=True)
        final_df.insert(0, 'question_id', [f"Q{i+1}" for i in range(len(final_df))])
        
        return final_df
    
    return pd.DataFrame()


def print_coverage_analysis(df: pd.DataFrame):
    """Print coverage analysis"""
    print("\n📊 COVERAGE ANALYSIS")
    print("="*70)
    print(f"\nTotal questions: {len(df)}")
    print(f"RAGAS: {len(df[df['generation_method'] == 'ragas'])}")
    print(f"Custom: {len(df[df['generation_method'] == 'custom'])}")
    print(f"\nQuestion types: {df['question_type'].nunique()} unique")
    print(df['question_type'].value_counts().head(10))
    print(f"\nComplexity: {df['complexity_level'].value_counts().to_dict()}")
    print(f"Reasoning: {df['reasoning_requirement'].value_counts().to_dict()}")


def print_sample_questions(df: pd.DataFrame):
    """Print sample questions"""
    print("\n📋 SAMPLE QUESTIONS")
    print("="*80)
    
    ragas_samples = df[df['generation_method'] == 'ragas'].head(2)
    custom_samples = df[df['generation_method'] == 'custom'].head(2)
    
    print("\n🔷 RAGAS QUESTIONS:\n")
    for _, row in ragas_samples.iterrows():
        print(f"Q: {row['question']}")
        print(f"Type: {row['question_type']}")
        print(f"A: {row['answer'][:150]}...\n")
    
    print("\n🔶 CUSTOM QUESTIONS:\n")
    for _, row in custom_samples.iterrows():
        print(f"Q: {row['question']}")
        print(f"Type: {row['question_type']}")
        print(f"Conv: {row['conversation_type']}")
        print(f"A: {row['answer'][:150]}...\n")


def export_datasets(df: pd.DataFrame):
    """Export datasets to files"""
    OUTPUT_DIR = "./outputs"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Full dataset
    full_file = f"{OUTPUT_DIR}/testset_hybrid_full.csv"
    df.to_csv(full_file, index=False)
    print(f"\n✅ Exported: {full_file}")
    
    # Simple format
    simple_df = df[['question', 'answer', 'source_file']].copy()
    simple_df.columns = ['question', 'ground_truth', 'source']
    simple_file = f"{OUTPUT_DIR}/testset_hybrid_simple.csv"
    simple_df.to_csv(simple_file, index=False)
    print(f"✅ Exported: {simple_file}")
    
    # Summary
    summary = {
        'total_questions': len(df),
        'ragas_count': len(df[df['generation_method'] == 'ragas']),
        'custom_count': len(df[df['generation_method'] == 'custom']),
        'question_types': df['question_type'].value_counts().to_dict(),
        'complexity': df['complexity_level'].value_counts().to_dict(),
        'reasoning': df['reasoning_requirement'].value_counts().to_dict(),
    }
    
    summary_file = f"{OUTPUT_DIR}/testset_hybrid_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Exported: {summary_file}")


def create_visualizations(df: pd.DataFrame):
    """Create visualizations"""
    OUTPUT_DIR = "./outputs"
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # RAGAS vs Custom
    method_counts = df['generation_method'].value_counts()
    axes[0, 0].pie(method_counts.values, labels=method_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('RAGAS vs Custom Distribution')
    
    # Question types
    type_counts = df['question_type'].value_counts().head(8)
    axes[0, 1].barh(range(len(type_counts)), type_counts.values)
    axes[0, 1].set_yticks(range(len(type_counts)))
    axes[0, 1].set_yticklabels(type_counts.index)
    axes[0, 1].set_title('Question Types')
    axes[0, 1].invert_yaxis()
    
    # Complexity
    complexity_counts = df['complexity_level'].value_counts()
    axes[1, 0].bar(range(len(complexity_counts)), complexity_counts.values)
    axes[1, 0].set_xticks(range(len(complexity_counts)))
    axes[1, 0].set_xticklabels(complexity_counts.index)
    axes[1, 0].set_title('Complexity Distribution')
    
    # Reasoning
    reasoning_counts = df['reasoning_requirement'].value_counts()
    axes[1, 1].bar(range(len(reasoning_counts)), reasoning_counts.values)
    axes[1, 1].set_xticks(range(len(reasoning_counts)))
    axes[1, 1].set_xticklabels(reasoning_counts.index)
    axes[1, 1].set_title('Reasoning Requirements')
    
    plt.tight_layout()
    viz_file = f"{OUTPUT_DIR}/testset_hybrid_analysis.png"
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved: {viz_file}")
    plt.close()


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    print("="*70)
    print("RAG Evaluation - Hybrid Test Set Generation")
    print("With Pydantic Structured Outputs")
    print("="*70)
    main()
