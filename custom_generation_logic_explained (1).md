# Custom Question Generation Logic: Large vs Small Documents

## Overview

This document explains how the **Custom Question Generation** logic works differently for large documents (100 pages) versus small documents (2 pages), and why we need special handling for small documents.

---

## The Challenge

**Target:** Generate 25 custom questions per document (50% of 50 total questions)

**Problem:** What happens when a document has fewer chunks than the number of questions we need to generate?

---

## Scenario 1: 100-Page Document (Large Document)

### Step 1: Document Chunking

**Input:**
- Document: 100 pages
- Estimated words: ~50,000 words
- Chunk settings: `chunk_size=1500`, `chunk_overlap=300`

**Process:**
```
100 pages → Combine all text → Split into chunks with overlap
```

**Output:**
- **Total chunks created: ~200 chunks**

---

### Step 2: Spatial Division

**Process:**
```
Divide 200 chunks into 4 spatial regions (quarters)
```

**Result:**
- **Region 1 (Beginning):** Chunks 0-49 (50 chunks)
- **Region 2 (Early-Middle):** Chunks 50-99 (50 chunks)
- **Region 3 (Late-Middle):** Chunks 100-149 (50 chunks)
- **Region 4 (End):** Chunks 150-199 (50 chunks)

**Why?** Ensures questions cover the entire document from beginning to end.

---

### Step 3: Sampling for Custom Generation

**Target:** Need 25 chunks for 25 custom questions

**Process:**
```
Sample evenly from all 4 regions:
- Region 1: Pick 6 chunks → [5, 10, 15, 20, 25, 30]
- Region 2: Pick 6 chunks → [55, 60, 65, 70, 75, 80]
- Region 3: Pick 6 chunks → [105, 110, 115, 120, 125, 130]
- Region 4: Pick 7 chunks → [155, 160, 165, 170, 175, 180, 185]
```

**Output:**
- **Sampled chunks: 25 chunks** (6+6+6+7)
- **Status: ✅ We have exactly what we need**

---

### Step 4: Question Type Assignment

**Process:**
```
Assign question types based on distribution:
- chatbot_style: 30% → 8 questions
- chatbot_memory: 10% → 3 questions
- scenario: 30% → 8 questions
- multi_hop: 15% → 4 questions
- comparative: 10% → 2 questions
- conditional: 5% → 1 question (rounded)
```

**Output:**
```
Chunk 5    → chatbot_style
Chunk 10   → scenario
Chunk 15   → chatbot_memory
Chunk 20   → multi_hop
Chunk 25   → chatbot_style
... (and so on for all 25 chunks)
```

**Result:** Each of the 25 chunks gets assigned one question type.

---

### Step 5: Question Generation

**Process:**
```
For each chunk:
  1. Read the chunk content
  2. Generate 1 question based on assigned type
  3. Generate answer from the chunk
  4. If marked for multi-turn, generate follow-up
```

**Example:**

**Chunk 5 (chatbot_style):**
```
Chunk text: "The compliance officer is responsible for monitoring 
regulatory requirements and ensuring all documentation is up to date..."

Generated Question: "Can you help me understand the compliance officer's 
main responsibilities?"

Answer: "The compliance officer is responsible for monitoring regulatory 
requirements and ensuring all documentation is up to date..."
```

**Chunk 15 (chatbot_memory with follow-up):**
```
Chunk text: "Component C produces output O and is configured in the 
config.yaml file under the production section..."

Turn 1 Question: "Which component produces output O?"
Turn 1 Answer: "Component C produces output O."

Turn 2 Follow-up: "Where is it configured?"
Turn 2 Answer: "It's configured in the config.yaml file under the 
production section."
```

**Output:**
- **25 initial questions** (one per chunk)
- **~10 follow-up questions** (40% of 25 get multi-turn)
- **Total: ~35 question turns**

---

### Step 6: Final Result for 100-Page Document

```
✅ 25 unique chunks used
✅ Each chunk used exactly once
✅ High diversity (25 different contexts)
✅ Questions spread across all 100 pages
✅ All regions covered (beginning, middle, end)
✅ No repetition of content
```

**Summary:**
- **Chunks needed:** 25
- **Chunks available:** 200
- **Strategy:** Pick best 25 chunks, use each once
- **Questions per chunk:** 1
- **Quality:** High (diverse contexts)

---

---

## Scenario 2: 2-Page Document (Small Document)

### Step 1: Document Chunking

**Input:**
- Document: 2 pages
- Estimated words: ~1,000 words
- Chunk settings: `chunk_size=1500`, `chunk_overlap=300`

**Process:**
```
2 pages → Combine all text → Try to split into chunks
```

**Output:**
- **Total chunks created: Only 3 chunks**
  - Chunk 0: First ~500 words
  - Chunk 1: Middle ~500 words (with overlap from Chunk 0)
  - Chunk 2: Last ~500 words (with overlap from Chunk 1)

**Why so few?** The entire document is only 1,000 words, which is less than the chunk_size of 1,500. With overlap, we get 3 small chunks.

---

### Step 2: Spatial Division

**Process:**
```
Divide 3 chunks into 4 spatial regions (quarters)
```

**Result:**
- **Region 1:** Chunk 0 (1 chunk)
- **Region 2:** Chunk 1 (1 chunk)
- **Region 3:** Chunk 2 (1 chunk)
- **Region 4:** (Empty - no chunks available)

**Problem:** We don't have enough chunks to fill all regions!

---

### Step 3: Sampling for Custom Generation

**Target:** Need 25 chunks for 25 custom questions

**Available:** Only 3 chunks exist

**Problem Identified:**
```
❌ Need: 25 chunks
❌ Have: 3 chunks
❌ Shortage: 22 chunks
```

**Current Logic (BROKEN - Without Fix):**
```
Sample from available chunks:
- Try to sample 25 chunks
- Only 3 available
- Can only sample 3 chunks maximum
```

**Output (BROKEN):**
- **Sampled chunks: 3 chunks** 
- **Status: ❌ We need 25 but only have 3!**

**Result with BROKEN logic:**
- Can only generate 3 questions (not 25) ❌

---

### Step 4: The FIX - Chunk Cycling

**NEW Logic (With Fix):**

Since we don't have enough unique chunks, we **cycle through** the available chunks multiple times:

```
Cycling Strategy:
Position 0  → Use Chunk 0
Position 1  → Use Chunk 1
Position 2  → Use Chunk 2
Position 3  → Use Chunk 0 (cycle back)
Position 4  → Use Chunk 1
Position 5  → Use Chunk 2
Position 6  → Use Chunk 0
Position 7  → Use Chunk 1
Position 8  → Use Chunk 2
... (continue until we have 25 positions)
```

**Cycling Pattern:**
```
Positions: [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
           │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │
Maps to:   C0 C1 C2 C0 C1 C2 C0 C1 C2 C0 C1 C2 C0 C1 C2 C0 C1 C2 C0 C1 C2 C0 C1 C2 C0
```

Where:
- C0 = Chunk 0
- C1 = Chunk 1
- C2 = Chunk 2

**Result:**
- **Chunk 0 used:** 9 times
- **Chunk 1 used:** 8 times
- **Chunk 2 used:** 8 times
- **Total positions:** 25 ✅

**Output (FIXED):**
- **Chunk positions: 25 positions** 
- **Status: ✅ We have 25 positions (chunks repeated)**

---

### Step 5: Question Type Assignment

**Process:**
```
Assign question types to all 25 positions:
- chatbot_style: 30% → 8 questions
- chatbot_memory: 10% → 3 questions
- scenario: 30% → 8 questions
- multi_hop: 15% → 4 questions
- comparative: 10% → 2 questions
- conditional: 5% → 1 question (rounded)
```

**Output:**
```
Position 0  (Chunk 0) → chatbot_style
Position 1  (Chunk 1) → scenario
Position 2  (Chunk 2) → chatbot_memory
Position 3  (Chunk 0) → multi_hop        ← Chunk 0 reused, different question type
Position 4  (Chunk 1) → chatbot_style    ← Chunk 1 reused, different question type
Position 5  (Chunk 2) → scenario         ← Chunk 2 reused, different question type
... (and so on for all 25 positions)
```

**Key Point:** Even though we're reusing chunks, we assign **different question types** each time, so the questions will be different!

---

### Step 6: Question Generation with Reused Chunks

**Process:**
```
For each position:
  1. Read the chunk content (may be repeated chunk)
  2. Generate 1 question based on assigned type (different type than before)
  3. Generate answer from the chunk
  4. If marked for multi-turn, generate follow-up
```

**Example - Same Chunk, Different Questions:**

**Chunk 0 (Used 9 times):**

```
Chunk text: "The escalation procedure requires immediate notification 
to the compliance officer within 2 hours of identifying any regulatory 
issue. The officer must then assess the severity and determine the 
appropriate response level..."
```

**Position 0 - chatbot_style:**
```
Question: "Can you help me understand the escalation procedure?"
Answer: "The escalation procedure requires immediate notification to 
the compliance officer within 2 hours..."
```

**Position 3 - multi_hop:**
```
Question: "If a regulatory issue is identified at 5 PM, what are the 
sequential steps that must happen, and what is the deadline for initial 
notification?"
Answer: "First, immediate notification to the compliance officer must 
occur within 2 hours (by 7 PM). Then, the officer must assess severity 
and determine response level..."
```

**Position 6 - scenario:**
```
Question: "A junior analyst discovers a potential compliance violation 
at 4:50 PM on Friday. The compliance officer is in a meeting until 6 PM. 
What should the analyst do?"
Answer: "According to the escalation procedure, immediate notification 
is required within 2 hours. The analyst should contact the compliance 
officer through the emergency notification system, as the 2-hour window 
cannot wait for the meeting to end..."
```

**Notice:** Same chunk content, but **3 completely different questions** based on different question types!

---

### Step 7: Final Result for 2-Page Document

```
✅ 25 questions generated
✅ Only 3 unique chunks available
✅ Each chunk reused ~8-9 times
⚠️  Lower diversity (only 3 different contexts)
⚠️  Same content generates multiple questions
✅ Questions still differ due to different question types
✅ All content covered (entire 2 pages)
```

**Summary:**
- **Chunks needed:** 25
- **Chunks available:** 3
- **Strategy:** Cycle through 3 chunks, use different question types
- **Questions per chunk:** 8-9 questions (from same chunk but different types)
- **Quality:** Medium (limited contexts, but varied question types)

---

---

## Side-by-Side Comparison

| Aspect | 100-Page Document | 2-Page Document |
|--------|-------------------|-----------------|
| **Total Pages** | 100 pages | 2 pages |
| **Total Words** | ~50,000 words | ~1,000 words |
| **Chunks Created** | ~200 chunks | 3 chunks |
| **Chunks Needed** | 25 chunks | 25 positions |
| **Chunks Available** | 200 chunks | 3 chunks |
| **Availability Status** | ✅ More than enough | ❌ Not enough |
| **Sampling Strategy** | Pick 25 from 200 | Cycle 3 chunks repeatedly |
| **Chunk Reuse** | None (each used once) | Heavy (each used 8-9 times) |
| **Questions per Chunk** | 1 question | 8-9 questions |
| **Question Diversity** | High (25 different contexts) | Medium (3 contexts, varied types) |
| **Content Coverage** | All 100 pages covered | All 2 pages covered |
| **Spatial Distribution** | Perfect (4 regions filled) | Limited (only 3 regions possible) |
| **Quality** | Excellent | Good (despite limitations) |

---

## Visual Representation

### 100-Page Document Flow

```
Document (100 pages)
    ↓
[Chunk Chunk Chunk Chunk ... Chunk Chunk Chunk]  ← 200 chunks
    ↓
Sample 25 chunks evenly
    ↓
[Chunk₅ Chunk₁₀ Chunk₁₅ ... Chunk₁₈₅]  ← 25 unique chunks
    ↓
Generate 1 question per chunk
    ↓
[Q₁ Q₂ Q₃ Q₄ Q₅ ... Q₂₅]  ← 25 questions
```

**Result:** High diversity, all unique chunks, excellent coverage

---

### 2-Page Document Flow (WITH FIX)

```
Document (2 pages)
    ↓
[Chunk₀ Chunk₁ Chunk₂]  ← Only 3 chunks
    ↓
Cycle chunks to create 25 positions
    ↓
[C₀ C₁ C₂ C₀ C₁ C₂ C₀ C₁ C₂ C₀ C₁ C₂ C₀ C₁ C₂ C₀ C₁ C₂ C₀ C₁ C₂ C₀ C₁ C₂ C₀]
    ↓
Assign different question types to each position
    ↓
[chatbot scenario memory multi chatbot scenario ...]  ← 25 different types
    ↓
Generate 1 question per position (different type = different question)
    ↓
[Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ Q₇ Q₈ Q₉ ... Q₂₅]  ← 25 questions
```

**Result:** Lower diversity (only 3 chunks), but still 25 varied questions due to different question types

---

## The Magic: Question Type Variation

### How Same Chunk Generates Different Questions

**Chunk Content:**
```
"The meeting room booking system requires advance approval from the 
department head for external participants. All bookings must be made 
at least 48 hours in advance. Emergency bookings can be approved by 
the compliance officer within 24 hours."
```

**Different Question Types Generate Different Questions:**

1. **chatbot_style:**
   - "Can you explain the meeting room booking procedure?"

2. **scenario:**
   - "A manager needs to book a meeting room for external clients tomorrow. The department head is on vacation. What are the available options?"

3. **multi_hop:**
   - "What are the approval requirements for booking a meeting room with external participants, and what alternative exists if the standard approval cannot be obtained?"

4. **comparative:**
   - "What's the difference between standard meeting room bookings and emergency bookings in terms of approval authority and timeline?"

5. **conditional:**
   - "If a department head is unavailable and an urgent external meeting is needed, what is the alternative approval process and what is the time constraint?"

6. **procedural:**
   - "What are the step-by-step requirements for booking a meeting room for external participants?"

7. **direct_factual:**
   - "How much advance notice is required for meeting room bookings?"

8. **compliance:**
   - "According to the booking policy, who has the authority to approve emergency bookings for external meetings?"

**Result:** 8 completely different questions from the same chunk content!

---

## Why This Approach Works

### For Large Documents (100 pages):

✅ **Natural behavior** - We have plenty of chunks, so we just pick the best ones
✅ **No special logic needed** - Standard sampling works perfectly
✅ **Maximum diversity** - Every question comes from different content
✅ **Optimal quality** - Each chunk provides unique context

### For Small Documents (2 pages):

✅ **Adaptive behavior** - We recognize the shortage and adapt
✅ **Chunk cycling** - Reuse available chunks strategically
✅ **Type variation** - Different question types create variety
✅ **Goal achievement** - Still generate target 25 questions
✅ **Content coverage** - All content (2 pages) is covered multiple times
⚠️  **Trade-off** - Less context diversity, but still valuable questions

---

## Key Insights

### 1. **Quantity is Maintained**
- 100-page doc → 25 questions ✅
- 2-page doc → 25 questions ✅

### 2. **Quality Adapts**
- 100-page doc → High diversity (25 unique contexts)
- 2-page doc → Medium diversity (3 contexts × different angles)

### 3. **Coverage is Guaranteed**
- 100-page doc → Questions from beginning, middle, and end
- 2-page doc → Questions from all available content (thoroughly covered)

### 4. **Question Types Drive Variation**
- Same content + Different question type = Different question
- This is the key to generating multiple questions from limited chunks

---

## Summary

### The Problem
Custom question generation needs a certain number of chunks. What happens when the document is too small to provide enough chunks?

### The Solution
**Chunk Cycling with Type Variation**

Instead of failing or generating fewer questions, we:
1. Recognize when chunks are insufficient
2. Cycle through available chunks multiple times
3. Assign different question types to each cycle
4. Generate unique questions from same content using different perspectives

### The Result
- ✅ Always generates target number of questions
- ✅ Works for any document size (2 pages to 1000 pages)
- ✅ Maintains quality through question type variation
- ✅ No hardcoded thresholds or assumptions
- ✅ Adaptive and dynamic approach

### The Trade-off
- Large docs: Maximum diversity from unique chunks
- Small docs: Lower diversity but compensated by varied question types

**Conclusion:** The system is robust, adaptive, and ensures consistent question generation regardless of document size!

---

## Implementation Note

This cycling logic needs to be explicitly implemented in the **Custom Question Generation** function. RAGAS handles this internally, but Custom generation requires the cycling strategy to be coded.

Without this fix:
- 100-page doc → 25 questions ✅
- 2-page doc → 3 questions ❌

With this fix:
- 100-page doc → 25 questions ✅
- 2-page doc → 25 questions ✅

---

---

## Summary Tables

### Summary Table: RAGAS vs Custom Behavior

| Scenario | RAGAS Behavior | Custom Behavior (Before Fix) | Custom Behavior (After Fix) |
|----------|----------------|------------------------------|----------------------------|
| **100 pages (200 chunks)** | ✅ Picks 25 chunks<br>✅ 1 Q per chunk<br>✅ 25 questions | ✅ Uses 25 chunks<br>✅ 1 Q per chunk<br>✅ 25 questions | ✅ Uses 25 chunks<br>✅ 1 Q per chunk<br>✅ 25 questions |
| **2 pages (3 chunks)** | ✅ Uses 3 chunks<br>✅ 8 Q per chunk<br>✅ 25 questions | ❌ Uses 3 chunks<br>❌ 1 Q per chunk<br>❌ **Only 3 questions** | ✅ Cycles 3 chunks<br>✅ Repeats chunks<br>✅ 25 questions |

---

### Side-by-Side Comparison: 100 Pages vs 2 Pages

| Aspect | 100 Pages | 2 Pages |
|--------|-----------|---------|
| **Chunks Created** | 200 | 3 |
| **Chunks per Region** | 50 | 0-1 |
| **Chunks Needed** | 50 (25+25) | 50 (25+25) |
| **Chunks Available** | 200 ✅ Plenty | 3 ❌ Not enough |
| **Sampling Logic** | Pick 50 from 200 | Use all 3 multiple times |
| **Questions per Chunk** | 1 question | 8-9 questions |
| **Diversity** | High (50 unique chunks) | Low (3 chunks repeated) |
| **Coverage** | All sections covered | All content covered but repetitive |
| **Spatial Works?** | ✅ Perfect | ⚠️ Limited (not enough regions) |

---

*Document Version: 1.0*
*Last Updated: January 2025*
