flowchart TD
    Start([Start: Fully Custom Question Generation]) --> Config[Load Configuration]
    
    Config --> ConfigDetails{Configuration Settings}
    ConfigDetails --> CD1[questions_per_document: 20]
    ConfigDetails --> CD2[num_spatial_regions: 4]
    ConfigDetails --> CD3[chunk_size: 1500]
    ConfigDetails --> CD4[multi_turn_ratio: 40%]
    
    CD1 --> InitLLM[Initialize LLMs]
    CD2 --> InitLLM
    CD3 --> InitLLM
    CD4 --> InitLLM
    
    InitLLM --> InitDetails[Base LLM + Structured LLMs]
    InitDetails --> LLM1[question_llm with GeneratedQuestion]
    InitDetails --> LLM2[followup_llm with FollowupQuestion]
    
    LLM1 --> LoadDocs[Load Documents from ./documents/]
    LLM2 --> LoadDocs
    
    LoadDocs --> FileTypes{Process File Types}
    
    FileTypes --> PDF[PDF Files: PyPDFLoader]
    FileTypes --> DOCX[DOCX Files: Docx2txtLoader]
    FileTypes --> TEXT[MD/TXT: read_text UTF-8]
    
    PDF --> DocsList[Combined Documents List]
    DOCX --> DocsList
    TEXT --> DocsList
    
    DocsList --> CheckDocs{Documents Found?}
    CheckDocs -->|No| EndError([End: No Documents])
    CheckDocs -->|Yes| GroupFiles[Groupby elements - Source]
    
    GroupFiles --> ChunkDocs[Chunk Documents]
    
    ChunkDocs --> ChunkProcess[RecursiveCharacterTextSplitter<br/>chunk_size=xx<br/>overlap=xx]
    
    ChunkProcess --> ChunkMeta[Add Chunk Metadata<br/>chunk_index, total_chunks]
    
    ChunkMeta --> SpatialSample[Spatial Sampling:<br/>4 regions]
    
    SpatialSample --> SpatialProcess{For Each Document}
    
    SpatialProcess --> DivideRegions[Divide into 4 Spatial Regions]
    DivideRegions --> SampleRegions[Sample Evenly from<br/>each Region]
    
    SampleRegions --> CheckChunks{Enough Chunks?}
    
    CheckChunks -->|Yes: n_chunks >= target| UseChunks[Use Sampled Chunks]
    CheckChunks -->|No: n_chunks < target| CycleChunks[Cycle Through Chunks<br/>Repeat: 0,1,2,0,1,2...]
    
    UseChunks --> GenPrep[Prepare Generation]
    CycleChunks --> GenPrep
    
    GenPrep --> TypeSeq[Create Question Type Sequence<br/>Based on Distribution]
    
    TypeSeq --> TypeDist{Question Types}
    TypeDist --> T1[chatbot_style: 10%]
    TypeDist --> T2[direct_factual: 10%]
    TypeDist --> T3[procedural: 15%]
    TypeDist --> T4[scenario: 20%]
    TypeDist --> T5[analytical: 10%]
    TypeDist --> T6[compliance: 5%]
    TypeDist --> T7[descriptive: 10%]
    TypeDist --> T8[multi_hop: 10%]
    TypeDist --> T9[comparative: 5%]
    TypeDist --> T10[conditional: 5%]
    
    T1 --> MultiTurn[Determine Multi-turn Questions<br/>40% get follow-ups]
    T2 --> MultiTurn
    T3 --> MultiTurn
    T4 --> MultiTurn
    T5 --> MultiTurn
    T6 --> MultiTurn
    T7 --> MultiTurn
    T8 --> MultiTurn
    T9 --> MultiTurn
    T10 --> MultiTurn
    
    MultiTurn --> GenLoop{For Each Chunk + Type}
    
    GenLoop --> GenQuestion[Generate Question with Pydantic]
    
    GenQuestion --> StructPrompt[Structured Prompt<br/>Context + Type + Requirements]
    
    StructPrompt --> LLMInvoke[question_llm.invoke]
    
    LLMInvoke --> PydanticReturn[Returns: GeneratedQuestion Object<br/>- question<br/>- answer<br/>- question_type<br/>- complexity<br/>- references]
    
    PydanticReturn --> AddMeta[Add Metadata<br/>source_file, region_id]
    
    AddMeta --> IsMultiTurn{Is Multi-turn?}
    
    IsMultiTurn -->|No| StoreSingle[Store as Single-turn]
    IsMultiTurn -->|Yes| GenFollowup[Generate Follow-up Questions]
    
    GenFollowup --> FollowupPrompt[Follow-up Prompt<br/>Context + Initial Q&A]
    
    FollowupPrompt --> FollowupLLM[followup_llm.invoke]
    
    FollowupLLM --> FollowupReturn[Returns: FollowupQuestion Object<br/>- followup_question<br/>- followup_answer<br/>- followup_type]
    
    FollowupReturn --> StoreMulti[Store as Multi-turn<br/>With Follow-ups]
    
    StoreSingle --> NextChunk{More Chunks?}
    StoreMulti --> NextChunk
    
    NextChunk -->|Yes| GenLoop
    NextChunk -->|No| EnsureCount[Ensure Exact Count<br/>Trim if > target]
    
    EnsureCount --> NextDoc{More Documents?}
    
    NextDoc -->|Yes| SpatialProcess
    NextDoc -->|No| FlattenData[Flatten to DataFrame]
    
    FlattenData --> FlatProcess[Process Each Question]
    
    FlatProcess --> InitialQ[Add Initial Question<br/>turn_number=1]
    InitialQ --> HasFollowup{Has Follow-ups?}
    
    HasFollowup -->|No| NextQFlat{More Questions?}
    HasFollowup -->|Yes| AddFollowups[Add Follow-up Rows<br/>turn_number=2,3...<br/>parent_question_id]
    
    AddFollowups --> NextQFlat
    NextQFlat -->|Yes| FlatProcess
    NextQFlat -->|No| CreateDF[Create Final DataFrame]
    
    CreateDF --> DFCols[Columns:<br/>question_id, question, answer<br/>source_file, region_id<br/>question_type, complexity<br/>conversation_type, turn_number<br/>is_followup, parent_question_id]
    
    DFCols --> Analysis[Coverage Analysis]
    
    Analysis --> Stats[Calculate Statistics<br/>- Total questions<br/>- Complexity distribution<br/>- Multiturn/simple count]
    
    Stats --> Export[Export Results]
    
    Export --> E1[testset_custom_full.csv<br/>Complete dataset]
    Export --> E4[testset_summary.json<br/>Statistics]
    







    ##################### Diagram for Followup and muti-turn generation #####################

    flowchart TD
    Start([Question Generation]) --> D1{Documents Loaded?}
    
    D1 -->|No| E1([Error: Add files])
    D1 -->|Yes| D2{Chunks Available?}
    
    D2 -->|No| E2([Error: Chunking failed])
    D2 -->|Yes| D3{n_chunks vs target?}
    
    D3 -->|n_chunks >= target| S1[Strategy: Direct sampling<br/>Use each chunk once]
    D3 -->|n_chunks < target| S2[Strategy: Chunk cycling<br/>Reuse chunks]
    
    S1 --> D4{Question Generated?}
    S2 --> D4
    
    D4 -->|No - Error| Skip[Skip this position]
    D4 -->|Yes| D5{Is Multi-turn Position?}
    
    D5 -->|No| Single[Single-turn]
    D5 -->|Yes| D6{Follow-up Generated?}
    
    D6 -->|No - Error| Single
    D6 -->|Yes| Multi[Multi-turn with follow-ups]
    
    Single --> D7{Count = Target?}
    Multi --> D7
    Skip --> D7
    
    D7 -->|< target| Continue[Continue generating]
    D7 -->|= target| Perfect[✅ Perfect count]
    D7 -->|> target| Trim[Trim to target]
    
    Continue --> D4
    Perfect --> Output[Export files]
    Trim --> Output
    
    Output --> End([Complete])
    
    style Start fill:#90EE90
    style End fill:#90EE90
    style E1 fill:#FFB6C1
    style E2 fill:#FFB6C1
    style S1 fill:#87CEEB
    style S2 fill:#FFD700
    style Perfect fill:#90EE90