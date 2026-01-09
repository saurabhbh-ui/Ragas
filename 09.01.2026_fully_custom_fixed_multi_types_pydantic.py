
"""
1. We need to add explaination for medium, easy, hard complexity levels in the QUESTION_GENERATION_PROMPT.
Example of single hop, 2-3 times hop and multiple hop. 

2. Explain Followup types : "clarification", "edge_case", "deeper_detail", "consequence", "exception"
"""



"""
RAG Evaluation - Fully Custom Test Set Generation
Pure Custom Generation (No RAGAS)

Pipeline:
1. Spatial division (4 regions) + Recursive chunking
2. Custom question generation (all types)
3. Multi-turn conversation generation
4. Automatic classification
5. Quality filtering
6. Coverage analysis & export
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
from langchain_openai import AzureChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document

# Visualization
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

print("✅ All imports successful")


# =====================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# =====================================================================

class GeneratedQuestion(BaseModel):
    """Structured output for generated questions"""
    question: str = Field(description="The generated question")
    answer: str = Field(description="Complete, detailed answer based on the context")
    question_type: str = Field(description="Type of question generated")
    complexity: Literal["easy", "medium", "hard"] = Field(
        description="Complexity level of the question. Complexity: easy (single-hop), medium (2-3 hops), hard (multi-hop synthesis)"
    )
    references: List[str] = Field(
        default_factory=list,
        description="List of references like 'Section X', 'Page Y'"
    )


class FollowupQuestion(BaseModel):
    """Structured output for follow-up questions"""
    followup_question: str = Field(description="The follow-up question")
    followup_answer: str = Field(description="Complete answer to the follow-up")
    followup_type: Literal[
        "clarification", "edge_case", "deeper_detail", "consequence", "exception"
    ] = Field(description="Type of follow-up question")


# =====================================================================
# CONFIGURATION
# =====================================================================

@dataclass
class CustomConfig:
    """Fully Custom Generation Configuration"""
    
    # Azure OpenAI Settings
    azure_endpoint: str = "https://your-endpoint.openai.azure.com/"
    azure_api_key: str = "your-api-key"
    azure_api_version: str = "2024-02-01"
    azure_deployment_gpt4: str = "gpt-4"
    
    # Coverage Strategy
    num_spatial_regions: int = 4
    questions_per_document: int = 20  # Total questions to generate
    
    # Chunking
    chunk_size: int = 1500
    chunk_overlap: int = 300
    
    # Question Type Distribution
    question_type_distribution: Dict[str, float] = field(default_factory=lambda: {
        "chatbot_style": 0.10,
        "direct_factual": 0.10,
        "procedural": 0.15,
        "scenario": 0.20,
        "analytical": 0.10,
        "compliance": 0.05,
        "descriptive": 0.10,
        "multi_hop": 0.10,
        "comparative": 0.05,
        "conditional": 0.05,
    })
    
    # Conversation Type Distribution
    single_turn_ratio: float = 0.60
    multi_turn_ratio: float = 0.40
    max_turns_per_conversation: int = 3
    
    # Domain
    domain_name: str = "Banking and Financial Services (BIS)"
    domain_context: str = "BIS Meeting Services, regulatory compliance, operational procedures"
    
    # LLM Parameters
    temperature: float = 0.7
    max_tokens: int = 3000
    
    # Quality
    min_quality_score: float = 7.0

config = CustomConfig()

print("✅ Configuration loaded")
print(f"   Questions per document: {config.questions_per_document}")
print(f"   Multi-turn ratio: {config.multi_turn_ratio * 100}%")
print(f"   Question types: {len(config.question_type_distribution)}")


# UPDATE YOUR AZURE CREDENTIALS HERE
config.azure_endpoint = os.getenv("AZURE_OAI_ENDPOINT") or "https://your-endpoint.openai.azure.com/"
config.azure_api_key = os.getenv("AZURE_OAI_API_KEY") or "your-api-key"
config.azure_deployment_gpt4 = os.getenv("AZURE_OAI_DEPLOYMENT") or "gpt-4"
config.azure_api_version = os.getenv("AZURE_OAI_API_VERSION") or "2024-02-01"

print("✅ Credentials configured")


# =====================================================================
# INITIALIZE LLM
# =====================================================================

# Base LLM
base_llm = AzureChatOpenAI(
    azure_endpoint=config.azure_endpoint,
    api_key=config.azure_api_key,
    api_version=config.azure_api_version,
    deployment_name=config.azure_deployment_gpt4,
    temperature=config.temperature,
    max_tokens=config.max_tokens,
)

# Structured LLMs with Pydantic models
question_llm = base_llm.with_structured_output(GeneratedQuestion)
followup_llm = base_llm.with_structured_output(FollowupQuestion)

# Test
test_response = base_llm.invoke("Say 'Ready'")
print(f"✅ LLM Test: {test_response.content}")
print("✅ Structured output LLMs initialized")


# =====================================================================
# DOCUMENT LOADING & CHUNKING
# =====================================================================

DOCUMENTS_DIR = "./documents"
Path(DOCUMENTS_DIR).mkdir(parents=True, exist_ok=True)


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


documents = load_documents(DOCUMENTS_DIR)

print(f"\n✅ Total documents loaded: {len(documents)}")

if len(documents) == 0:
    print("⚠️  No documents found!")
    print(f"   Please add files to: {DOCUMENTS_DIR}")
    print(f"   Supported formats: PDF, DOCX, MD, TXT")
else:
    # Group by source file and file type
    files_by_type = {}
    for doc in documents:
        source = doc.metadata.get("source_file", "unknown")
        file_type = doc.metadata.get("file_type", "unknown")
        
        if file_type not in files_by_type:
            files_by_type[file_type] = {}
        
        if source not in files_by_type[file_type]:
            files_by_type[file_type][source] = 0
        
        files_by_type[file_type][source] += 1
    
    print("\n📊 Documents loaded by type:")
    for file_type, files in files_by_type.items():
        print(f"\n  {file_type.upper()}:")
        for file, count in files.items():
            print(f"    - {file}: {count} pages/sections")


def chunk_documents(documents: List[Document], config: CustomConfig) -> Dict[str, List[Document]]:
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


if documents:
    chunked_docs_by_file = chunk_documents(documents, config)
    print(f"\n✅ Chunking complete")
else:
    chunked_docs_by_file = {}
    print("⚠️  No documents to chunk")


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


# Sample chunks from each document
sampled_chunks_by_file = {}
for source, chunks in chunked_docs_by_file.items():
    sampled = sample_chunks_spatial(chunks, config.questions_per_document, config.num_spatial_regions)
    sampled_chunks_by_file[source] = sampled
    print(f"{source}: Sampled {len(sampled)} chunks")

print(f"\n✅ Sampling complete")


# =====================================================================
# QUESTION GENERATION PROMPTS
# =====================================================================

QUESTION_GENERATION_PROMPT = """
You are a professional test question generator for {domain_name}.

DOCUMENT CONTEXT:
{context}

QUESTION TYPE: {question_type}

QUESTION TYPE DEFINITIONS:

1. **chatbot_style**: Conversational, help-seeking questions
   Example: "Can you help me understand the meeting room booking policy?"

2. **direct_factual**: Direct, specific factual questions
   Example: "What is the escalation timeframe for Priority 1 incidents?"

3. **procedural**: Step-by-step process questions
   Example: "What are the steps to book a meeting room for external participants?"

4. **scenario**: Realistic business situation questions
   Example: "A client requests an urgent meeting room for tomorrow at 8 AM, but all rooms are booked. What should the coordinator do?"

5. **analytical**: Analysis, evaluation, or comparison questions
   Example: "Analyze the differences between Priority 1 and Priority 2 incident response protocols."

6. **compliance**: Regulatory or policy questions
   Example: "According to MiFID II requirements, what documentation must be retained?"

7. **descriptive**: Questions requiring detailed descriptions
   Example: "Describe the roles and responsibilities of the compliance officer."

8. **multi_hop**: Questions requiring multiple pieces of information
   Example: "If a Priority 1 incident occurs during non-business hours and the primary contact is unavailable, what is the backup procedure?"

9. **comparative**: Questions comparing two or more items
   Example: "Compare the booking procedures for internal meetings versus external meetings."

10. **conditional**: Questions with if-then scenarios
    Example: "If a meeting room booking is cancelled less than 24 hours in advance, what are the consequences?"

QUALITY REQUIREMENTS:
- Be SPECIFIC: Reference sections, timeframes, roles, procedures
- Use PROFESSIONAL language appropriate for the type
- Make ANSWERABLE from the document context provided
- Avoid GENERIC questions that could apply to any document

Generate ONE {question_type} question with its complete answer, complexity level, and any references.
"""

print("✅ Question generation prompt defined")


FOLLOWUP_GENERATION_PROMPT = """
You are generating a FOLLOW-UP question for a multi-turn conversation.

DOCUMENT CONTEXT:
{context}

INITIAL QUESTION:
{initial_question}

INITIAL ANSWER:
{initial_answer}

FOLLOW-UP REQUIREMENTS:
1. **Build on the initial question** - Reference or extend it naturally
2. **Dig deeper** - Ask for more detail, exceptions, edge cases, or consequences
3. **Maintain context** - Assume the initial answer is known
4. **Be conversational** - Natural progression of the conversation

FOLLOW-UP TYPES:
- **clarification**: "In the procedure you mentioned, what happens if X?"
- **edge_case**: "What if the standard approach doesn't apply because Y?"
- **deeper_detail**: "For the documentation requirement, how long must records be retained?"
- **consequence**: "After following that procedure, what are the next steps?"
- **exception**: "Are there any circumstances where this rule can be waived?"

Generate ONE follow-up question with its answer and type.
"""

print("✅ Follow-up generation prompt defined")


# =====================================================================
# QUESTION GENERATION FUNCTIONS
# =====================================================================

def generate_question(chunk: Document, question_type: str, config: CustomConfig) -> Optional[Dict]:
    """Generate a single question using Pydantic structured output"""
    prompt = QUESTION_GENERATION_PROMPT.format(
        context=chunk.page_content[:2500],
        question_type=question_type,
        domain_name=config.domain_name
    )
    
    try:
        # Use structured output - returns Pydantic model directly
        result: GeneratedQuestion = question_llm.invoke(prompt)
        
        # Convert to dict and add metadata
        qa_dict = {
            'question': result.question,
            'answer': result.answer,
            'question_type': result.question_type,
            'complexity': result.complexity,
            'references': result.references,
            'source_file': chunk.metadata.get('source_file', 'unknown'),
            'region_id': chunk.metadata.get('region_id', 0)
        }
        
        return qa_dict
    except Exception as e:
        print(f"  ⚠️  Error generating question: {str(e)[:80]}")
        return None


def generate_followup(initial_qa: Dict, chunk: Document) -> Optional[Dict]:
    """Generate a follow-up question using Pydantic structured output"""
    prompt = FOLLOWUP_GENERATION_PROMPT.format(
        context=chunk.page_content[:2500],
        initial_question=initial_qa['question'],
        initial_answer=initial_qa['answer']
    )
    
    try:
        # Use structured output - returns Pydantic model directly
        result: FollowupQuestion = followup_llm.invoke(prompt)
        
        # Convert to dict
        followup_dict = {
            'followup_question': result.followup_question,
            'followup_answer': result.followup_answer,
            'followup_type': result.followup_type
        }
        
        return followup_dict
    except Exception as e:
        print(f"  ⚠️  Error generating follow-up: {str(e)[:60]}")
        return None


def get_question_type_sequence(n_questions: int, distribution: Dict[str, float]) -> List[str]:
    """Create question type sequence based on distribution"""
    sequence = []
    for q_type, ratio in distribution.items():
        count = max(1, int(n_questions * ratio))
        sequence.extend([q_type] * count)
    
    import random
    random.shuffle(sequence)
    return sequence[:n_questions]


print("✅ Ready to generate questions")


# =====================================================================
# MAIN QUESTION GENERATION
# =====================================================================

all_questions = []

for source_file, sampled_chunks in sampled_chunks_by_file.items():
    print(f"\n{'='*70}")
    print(f"Generating questions for: {source_file}")
    print(f"{'='*70}")
    
    n_chunks = len(sampled_chunks)
    target_questions = config.questions_per_document
    
    # HANDLE SMALL DOCUMENTS: Cycle chunks if not enough
    if n_chunks < target_questions:
        print(f"  ⚠️  Only {n_chunks} chunks available, need {target_questions} questions")
        print(f"  Cycling through chunks...")
        
        # Cycle chunks to match target
        chunks_to_use = []
        for i in range(target_questions):
            chunk_idx = i % n_chunks
            chunks_to_use.append(sampled_chunks[chunk_idx])
        
        print(f"  Using {len(chunks_to_use)} chunk positions (from {n_chunks} unique chunks)")
    else:
        chunks_to_use = sampled_chunks[:target_questions]
    
    # Get question type sequence
    question_types = get_question_type_sequence(target_questions, config.question_type_distribution)
    
    print(f"\nQuestion type breakdown:")
    for q_type in set(question_types):
        count = question_types.count(q_type)
        print(f"  - {q_type}: {count}")
    
    # Decide which questions get follow-ups (multi-turn)
    n_multiturn = int(len(question_types) * config.multi_turn_ratio)
    multiturn_indices = np.random.choice(len(question_types), n_multiturn, replace=False)
    
    doc_questions = []
    
    for idx, (chunk, q_type) in enumerate(tqdm(list(zip(chunks_to_use, question_types)), 
                                                desc="Generating")):
        # Generate initial question
        qa = generate_question(chunk, q_type, config)
        
        if qa:
            qa['conversation_type'] = 'multi_turn' if idx in multiturn_indices else 'single_turn'
            qa['turn_number'] = 1
            qa['parent_question_id'] = None
            qa['has_followup'] = False
            qa['followup_questions'] = []
            
            doc_questions.append(qa)
            
            # Generate follow-ups if multi-turn
            if idx in multiturn_indices:
                n_turns = np.random.randint(1, config.max_turns_per_conversation)
                followups = []
                
                for turn in range(n_turns):
                    followup = generate_followup(qa, chunk)
                    if followup:
                        followups.append(followup)
                
                if followups:
                    qa['has_followup'] = True
                    qa['followup_questions'] = followups
    
    # ENSURE EXACT COUNT: Trim if over (shouldn't happen but safety check)
    if len(doc_questions) > target_questions:
        print(f"  ⚠️  Generated {len(doc_questions)}, trimming to {target_questions}")
        doc_questions = doc_questions[:target_questions]
    
    all_questions.extend(doc_questions)
    
    print(f"\n✅ Generated {len(doc_questions)} questions (target: {target_questions})")
    print(f"   - Single-turn: {sum(1 for q in doc_questions if q['conversation_type'] == 'single_turn')}")
    print(f"   - Multi-turn: {sum(1 for q in doc_questions if q['conversation_type'] == 'multi_turn')}")
    print(f"   - Total with follow-ups: {sum(len(q['followup_questions']) for q in doc_questions)}")

print(f"\n{'='*70}")
print(f"✅ TOTAL QUESTIONS GENERATED: {len(all_questions)}")
print(f"   - Multi-turn conversations: {sum(1 for q in all_questions if q['has_followup'])}")
print(f"   - Total follow-up questions: {sum(len(q['followup_questions']) for q in all_questions)}")
print(f"{'='*70}")


# =====================================================================
# CREATE DATAFRAME
# =====================================================================

# Flatten questions to DataFrame format
flat_questions = []

for qa in all_questions:
    # Initial question
    flat_questions.append({
        'question_id': f"Q{len(flat_questions) + 1}",
        'question': qa['question'],
        'answer': qa['answer'],
        'source_file': qa.get('source_file', 'unknown'),
        'region_id': qa.get('region_id', 0) + 1,
        'question_type': qa.get('question_type', 'unknown'),
        'complexity': qa.get('complexity', 'medium'),
        'conversation_type': qa.get('conversation_type', 'single_turn'),
        'turn_number': 1,
        'parent_question_id': None,
        'is_followup': False,
        'references': ', '.join(qa.get('references', [])) if qa.get('references') else ''
    })
    
    # Follow-up questions
    if qa.get('has_followup'):
        parent_id = flat_questions[-1]['question_id']
        for i, followup in enumerate(qa.get('followup_questions', []), start=2):
            flat_questions.append({
                'question_id': f"Q{len(flat_questions) + 1}",
                'question': followup['followup_question'],
                'answer': followup['followup_answer'],
                'source_file': qa.get('source_file', 'unknown'),
                'region_id': qa.get('region_id', 0) + 1,
                'question_type': 'followup',
                'complexity': qa.get('complexity', 'medium'),
                'conversation_type': 'multi_turn',
                'turn_number': i,
                'parent_question_id': parent_id,
                'is_followup': True,
                'references': ''
            })

final_df = pd.DataFrame(flat_questions)

print(f"\n✅ Created DataFrame with {len(final_df)} total questions (including follow-ups)")


# =====================================================================
# COVERAGE ANALYSIS
# =====================================================================

if len(final_df) > 0:
    print("\n📊 COVERAGE ANALYSIS")
    print("="*70)
    
    print(f"\nTotal questions: {len(final_df)}")
    print(f"Initial questions: {len(final_df[~final_df['is_followup']])}")
    print(f"Follow-up questions: {len(final_df[final_df['is_followup']])}")
    
    print(f"\nQuestion type distribution:")
    print(final_df['question_type'].value_counts())
    
    print(f"\nComplexity distribution:")
    print(final_df['complexity'].value_counts())
    
    print(f"\nConversation type:")
    print(final_df['conversation_type'].value_counts())
    
    print(f"\nBy source file:")
    print(final_df.groupby('source_file').size())


# =====================================================================
# SAMPLE QUESTIONS REVIEW
# =====================================================================

print("\n📋 SAMPLE GENERATED QUESTIONS\n")
print("="*80)

sample_size = min(5, len(final_df))
for i, row in final_df.head(sample_size).iterrows():
    print(f"\n🔷 QUESTION {i+1}")
    print(f"   ID: {row['question_id']}")
    print(f"   Type: {row['question_type'].upper()}")
    print(f"   Conversation: {row['conversation_type'].upper()}")
    print(f"   Source: {row['source_file']}")
    print(f"   Region: {row['region_id']}")
    
    print(f"\n   Q: {row['question']}")
    answer_preview = row['answer'][:150] + "..." if len(row['answer']) > 150 else row['answer']
    print(f"   A: {answer_preview}")
    print(f"   Complexity: {row['complexity']}")
    
    if row['is_followup']:
        print(f"   Parent: {row['parent_question_id']}")
        print(f"   Turn: {row['turn_number']}")


# =====================================================================
# EXPORT
# =====================================================================

OUTPUT_DIR = "./outputs"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Full dataset
full_file = f"{OUTPUT_DIR}/testset_custom_full.csv"
final_df.to_csv(full_file, index=False)
print(f"\n✅ Exported: {full_file}")

# Simple format (for RAG evaluation)
simple_df = final_df[['question', 'answer', 'source_file']].copy()
simple_df.columns = ['question', 'ground_truth', 'source']
simple_file = f"{OUTPUT_DIR}/testset_custom_simple.csv"
simple_df.to_csv(simple_file, index=False)
print(f"✅ Exported: {simple_file}")

# Conversation chains (multi-turn only)
conversation_df = final_df[final_df['conversation_type'] == 'multi_turn'].copy()
if len(conversation_df) > 0:
    conv_file = f"{OUTPUT_DIR}/testset_conversation_chains.csv"
    conversation_df.to_csv(conv_file, index=False)
    print(f"✅ Exported: {conv_file}")

# Summary
summary = {
    'total_questions': len(final_df),
    'initial_questions': len(final_df[~final_df['is_followup']]),
    'followup_questions': len(final_df[final_df['is_followup']]),
    'question_types': final_df['question_type'].value_counts().to_dict(),
    'complexity': final_df['complexity'].value_counts().to_dict(),
    'conversation_types': final_df['conversation_type'].value_counts().to_dict(),
}

summary_file = f"{OUTPUT_DIR}/testset_summary.json"
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✅ Exported: {summary_file}")


# =====================================================================
# VISUALIZATION
# =====================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Question types
type_counts = final_df['question_type'].value_counts()
axes[0, 0].barh(range(len(type_counts)), type_counts.values)
axes[0, 0].set_yticks(range(len(type_counts)))
axes[0, 0].set_yticklabels(type_counts.index)
axes[0, 0].set_title('Question Type Distribution')
axes[0, 0].invert_yaxis()

# Conversation types
conv_counts = final_df['conversation_type'].value_counts()
axes[0, 1].pie(conv_counts.values, labels=conv_counts.index, autopct='%1.1f%%')
axes[0, 1].set_title('Single-turn vs Multi-turn')

# Complexity
complexity_counts = final_df['complexity'].value_counts()
axes[1, 0].bar(range(len(complexity_counts)), complexity_counts.values)
axes[1, 0].set_xticks(range(len(complexity_counts)))
axes[1, 0].set_xticklabels(complexity_counts.index)
axes[1, 0].set_title('Complexity Distribution')

# Questions by source
source_counts = final_df.groupby('source_file').size()
axes[1, 1].bar(range(len(source_counts)), source_counts.values)
axes[1, 1].set_xticks(range(len(source_counts)))
axes[1, 1].set_xticklabels(source_counts.index, rotation=45, ha='right')
axes[1, 1].set_title('Questions by Source File')

plt.tight_layout()
viz_file = f"{OUTPUT_DIR}/testset_analysis.png"
plt.savefig(viz_file, dpi=300, bbox_inches='tight')
print(f"✅ Visualization saved: {viz_file}")
plt.close()

print("\n" + "="*70)
print("✅ COMPLETE! All files saved to ./outputs/")
print("="*70)
