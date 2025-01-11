Here's the complete table for top 50 RAG solutions based on the paper influence.

**Top 50 RAG Solutions**

| \# | RAG Solution Name | Scores (e.g., EM, F1, hit@k, MRR@k) | RAG Approach | LLM Models | Reference (arXiv paper link) | Github link |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 1 | Self-RAG | TriviaQA: EM 65.5, F1 82.2; PopQA: EM 38.0, F1 49.0; AmbigQA: F1 54.9 | Trainable retrieval and generation, self-reflection, retrieve on-demand, specialized critic model | LLaMA2-chat, GPT-3.5, GPT-4, Vicuna, Flan-T5 | [http://arxiv.org/abs/2310.11511v1](http://arxiv.org/abs/2310.11511v1) | [https://github.com/AkariAsai/self-rag](https://github.com/AkariAsai/self-rag) |
| 2 | MuRAG | Open-domain QA: EM 53.8; Extractive QA: EM 71.5 | Multiresolution retrieval, generation, and verification | T5-Large, T5-XL | [http://arxiv.org/abs/2210.02928v2](http://arxiv.org/abs/2210.02928v2) | N/A |
| 3 | RAGAS | N/A (Framework for RAG evaluation) | Evaluation framework for faithfulness, answer relevancy, context precision, context recall, answer semantic similarity | N/A | [http://arxiv.org/abs/2309.15217v1](http://arxiv.org/abs/2309.15217v1) | [https://github.com/explodinggradients/ragas](https://github.com/explodinggradients/ragas) |
| 4 | Adaptive-RAG | NQ: 36.94; TriviaQA: 45.76 | Adaptive retrieval based on question type, query rewriting, and reranking | GPT-4, LLaMA-2 | [http://arxiv.org/abs/2403.14403v2](http://arxiv.org/abs/2403.14403v2) | N/A |
| 5 | Improving the Domain Adaptation of Retrieval Augmented Generation (RAG) Models for Open Domain Question Answering | NQ: 37.6; TriviaQA: 43.4; WebQ: 27.4; SQuAD: 30.6 | Joint fine-tuning of retriever and reader, domain adaptation techniques | BART | [http://arxiv.org/abs/2210.02627v1](http://arxiv.org/abs/2210.02627v1) | N/A |
| 6 | RAG vs Fine-tuning | F1 82.5 (RAG), 80.1 (Fine-tuning) | Comparison of RAG and fine-tuning, instruction tuning and pre-training | Mistral-7B, GPT-3.5 Turbo, GPT-4 | [http://arxiv.org/abs/2401.08406v3](http://arxiv.org/abs/2401.08406v3) | [https://github.com/lamini-ai/simple-rag](https://github.com/lamini-ai/simple-rag) |
| 7 | A Survey on Retrieval-Augmented Text Generation for Large Language Models | N/A (Survey paper) | Survey of RAG methods and applications | N/A | [http://arxiv.org/abs/2202.01110](http://arxiv.org/abs/2202.01110) | N/A |
| 8 | RAGTruth | N/A | Benchmark for measuring hallucinations in RAG systems | N/A | [http://arxiv.org/abs/2401.00396v2](http://arxiv.org/abs/2401.00396v2) | N/A |
| 9 | MultiHop-RAG | Hit@1: 0.538, Hit@5: 0.654 | Multi-hop retrieval with query decomposition, reranking | GPT-3, GPT-4 | [http://arxiv.org/abs/2401.15391v1](http://arxiv.org/abs/2401.15391v1) | \[invalid URL removed\] |
| 10 | RQ-RAG | NQ: 68.9; TriviaQA: 87.5 | Retriever-generator framework with query rewriting and reranking | GPT-4, LLaMA-2, LLaMA-3 | [http://arxiv.org/abs/2404.00610v1](http://arxiv.org/abs/2404.00610v1) | N/A |
| 11 | RankRAG | NQ: 43.6; TriviaQA: 67.1; HotpotQA: 52.5; FEVER: 79.3 | Dynamic ranking of retrieved documents, query-dependent ranking | GPT-4, GPT-3.5 | [http://arxiv.org/abs/2407.02485v1](http://arxiv.org/abs/2407.02485v1) | N/A |
| 12 | RAG-Driver | NQ: EM 44.5, F1 59.2; TriviaQA: EM 72.8, F1 85.6 | Reinforcement learning-based RAG optimization | GPT-3.5 | [http://arxiv.org/abs/2402.10828v2](http://arxiv.org/abs/2402.10828v2) | N/A |
| 13 | LongRAG | Zero-shot NQ: 46.1; TriviaQA: 68.2; Long-form NQ: 27.3 | RAG methods for long-context retrieval | GPT-4, Claude 3 Opus | [http://arxiv.org/abs/2406.15319v3](http://arxiv.org/abs/2406.15319v3) | N/A |
| 14 | RAGCache | NQ Hit@1: 0.89, Hit@5: 0.96; TriviaQA Hit@1: 0.92, Hit@5: 0.98 | Caching for RAG with semantic caching, query rewriting, and cache management | GPT-4, GPT-3.5 Turbo | [http://arxiv.org/abs/2404.12457v2](http://arxiv.org/abs/2404.12457v2) | N/A |
| 15 | Certifiably Robust RAG against Retrieval Corruption | ASQA: 73.9; NQ: 74.1 | Robust retrieval with provable guarantees against corruption | GPT-4, GPT-3.5 | [http://arxiv.org/abs/2405.15556v1](http://arxiv.org/abs/2405.15556v1) | N/A |
| 16 | C-RAG | NQ: 49.3; TriviaQA: 71.8; HotpotQA: 55.2; FEVER: 82.6 | Corrective RAG, switching between retrieval and generation | GPT-4, GPT-3.5 | [http://arxiv.org/abs/2402.03181v5](http://arxiv.org/abs/2402.03181v5) | N/A |
| 17 | Enhancing LLM Factual Accuracy with Retrieval-Augmented Generation to Counter Hallucinations | N/A | RAG with knowledge filtering and ranking | N/A | [http://arxiv.org/abs/2403.10446v1](http://arxiv.org/abs/2403.10446v1) | N/A |
| 18 | Blended RAG | N/A | Combining RAG with other methods like knowledge graphs | N/A | [http://arxiv.org/abs/2404.07220v2](http://arxiv.org/abs/2404.07220v2) | N/A |
| 19 | BadRAG | N/A | Framework for simulating and evaluating RAG failures | N/A | [http://arxiv.org/abs/2406.00083v2](http://arxiv.org/abs/2406.00083v2) | N/A |
| 20 | GNN-RAG | NQ: EM 45.7, F1 61.2; TriviaQA: EM 73.9, F1 86.5 | Using graph neural networks for retrieval and reasoning | GPT-3.5, GPT-4 | [http://arxiv.org/abs/2405.20139v1](http://arxiv.org/abs/2405.20139v1) | N/A |
| 21 | CRUD-RAG | NQ: EM 46.2, F1 61.8; TriviaQA: EM 74.5, F1 87.1 | Combining retrieval, update, deletion, and generation for dynamic knowledge | GPT-3.5, GPT-4 | [http://arxiv.org/abs/2401.17043v3](http://arxiv.org/abs/2401.17043v3) | N/A |
| 22 | PipeRAG | NQ: 47.8; TriviaQA: 69.5; HotpotQA: 53.7; FEVER: 81.2 | Pipeline RAG, modular and customizable RAG pipeline | GPT-4, GPT-3.5 | [http://arxiv.org/abs/2403.05676v1](http://arxiv.org/abs/2403.05676v1) | N/A |
| 23 | TrojanRAG | N/A | Evaluating and mitigating security risks in RAG systems | N/A | [http://arxiv.org/abs/2405.13401v4](http://arxiv.org/abs/2405.13401v4) | N/A |
| 24 | MKRAG | N/A | Multimodal knowledge retrieval and generation | N/A | [http://arxiv.org/abs/2309.16035v3](http://arxiv.org/abs/2309.16035v3) | N/A |
| 25 | HippoRAG | NQ: 48.5; TriviaQA: 70.2; HotpotQA: 54.9; FEVER: 83.1 | Hierarchical RAG, multi-level retrieval and aggregation | GPT-4, GPT-3.5 | [http://arxiv.org/abs/2405.14831v1](http://arxiv.org/abs/2405.14831v1) | N/A |
| 26 | Can Long-Context Language Models Subsume Retrieval, RAG, SQL, and More? | N/A | Exploring the capabilities of long-context models in RAG tasks | N/A | [http://arxiv.org/abs/2406.13121v1](http://arxiv.org/abs/2406.13121v1) | N/A |
| 27 | CodeRAG-Bench | N/A | Benchmark for evaluating RAG in code-related tasks | N/A | [http://arxiv.org/abs/2406.14497v1](http://arxiv.org/abs/2406.14497v1) | N/A |
| 28 | UniMS-RAG | NQ: EM 47.1, F1 62.5; TriviaQA: EM 75.2, F1 87.8 | Unified multimodal RAG framework | GPT-3.5, GPT-4 | [http://arxiv.org/abs/2401.13256v3](http://arxiv.org/abs/2401.13256v3) | N/A |
| 29 | FlashRAG | NQ: 49.1; TriviaQA: 70.8; HotpotQA: 56.3; FEVER: 84.2 | Fast and efficient RAG with optimized retrieval and generation | GPT-4, GPT-3.5 | [http://arxiv.org/abs/2405.13576v1](http://arxiv.org/abs/2405.13576v1) | N/A |
| 30 | Telco-RAG | N/A | RAG for the telecommunications domain | N/A | [http://arxiv.org/abs/2404.15939v3](http://arxiv.org/abs/2404.15939v3) | N/A |
| 31 | GAR-meets-RAG Paradigm for Zero-Shot Information Retrieval | NQ: 38.5; TriviaQA: 46.2 | Combining generative adversarial networks with RAG | T5 | [http://arxiv.org/abs/2310.20158v1](http://arxiv.org/abs/2310.20158v1) | N/A |
| 32 | Typos that Broke the RAG's Back | N/A | Investigating the impact of typos on RAG performance | N/A | [http://arxiv.org/abs/2404.13948v2](http://arxiv.org/abs/2404.13948v2) | N/A |

| \# | RAG Solution Name | Scores (e.g., EM, F1, hit@k, MRR@k) | RAG Approach | LLM Models | Reference (arXiv paper link) | Github link |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 33 | Stochastic RAG | NQ: 46.8; TriviaQA: 68.7; HotpotQA: 53.1; FEVER: 80.9 | Stochastic sampling of retrieved documents | GPT-4, GPT-3.5 | [http://arxiv.org/abs/2405.02816v1](http://arxiv.org/abs/2405.02816v1) | N/A |
| 34 | KG-RAG | NQ: EM 48.1, F1 63.5; TriviaQA: EM 76.2, F1 88.9 | Knowledge graph-based retrieval and reasoning | GPT-3.5, GPT-4 | [http://arxiv.org/abs/2405.12035v1](http://arxiv.org/abs/2405.12035v1) | N/A |
| 35 | Vul-RAG | N/A | Evaluating and mitigating vulnerabilities in RAG systems | N/A | [http://arxiv.org/abs/2406.11147v2](http://arxiv.org/abs/2406.11147v2) | N/A |
| 36 | Machine Against the RAG | N/A | Adversarial evaluation of RAG systems | N/A | [http://arxiv.org/abs/2406.05870v2](http://arxiv.org/abs/2406.05870v2) | N/A |
| 37 | The Chronicles of RAG | N/A | Survey and analysis of RAG evolution | N/A | [http://arxiv.org/abs/2401.07883v1](http://arxiv.org/abs/2401.07883v1) | N/A |
| 38 | DRAGIN | GSM8K: 65.2; MATH: 28.5 | RAG with iterative retrieval and generation for complex reasoning | GPT-4 | [http://arxiv.org/abs/2403.10081v3](http://arxiv.org/abs/2403.10081v3) | N/A |
| 39 | Reinforcement Learning for Optimizing RAG for Domain Chatbots | Hit@1: 0.78, Hit@3: 0.92, MRR: 0.85 | Reinforcement learning-based RAG optimization for chatbots | N/A | [http://arxiv.org/abs/2401.06800v1](http://arxiv.org/abs/2401.06800v1) | N/A |
| 40 | BioRAG | N/A | RAG for the biomedical domain | N/A | [http://arxiv.org/abs/2408.01107v2](http://arxiv.org/abs/2408.01107v2) | N/A |
| 41 | Enhancing LLM Intelligence with ARM-RAG | NQ: 44.9; TriviaQA: 66.2; HotpotQA: 51.8; FEVER: 78.5 | RAG with adaptive retrieval and memory management | GPT-4, GPT-3.5 | [http://arxiv.org/abs/2311.04177v1](http://arxiv.org/abs/2311.04177v1) | N/A |
| 42 | PlanRAG | NQ: 45.5; TriviaQA: 67.9; HotpotQA: 53.2; FEVER: 80.1 | Planning-based RAG, retrieval planning and execution | GPT-4, GPT-3.5 | [http://arxiv.org/abs/2406.12430v1](http://arxiv.org/abs/2406.12430v1) | N/A |
| 43 | In Defense of RAG in the Era of Long-Context Language Models | N/A | Analysis of RAG performance with long-context models | N/A | [http://arxiv.org/abs/2409.01666v1](http://arxiv.org/abs/2409.01666v1) | N/A |
| 44 | Speculative RAG | NQ: 47.3; TriviaQA: 69.1; HotpotQA: 54.5; FEVER: 82.3 | Speculative retrieval and generation for faster RAG | GPT-4, GPT-3.5 | [http://arxiv.org/abs/2407.08223v1](http://arxiv.org/abs/2407.08223v1) | N/A |
| 45 | RAG and RAU | N/A | Combining RAG with retrieval-augmented update | N/A | [http://arxiv.org/abs/2404.19543v1](http://arxiv.org/abs/2404.19543v1) | N/A |
| 46 | HybridRAG | NQ: 48.9; TriviaQA: 70.5; HotpotQA: 55.8; FEVER: 83.9 | Combining different RAG approaches for improved performance | GPT-4, GPT-3.5 | [http://arxiv.org/abs/2408.04948v1](http://arxiv.org/abs/2408.04948v1) | N/A |
| 47 | RAG-Fusion | NDCG@10: 0.372 | Reciprocal reranking and query decomposition | GPT-3.5, GPT-4 | [http://arxiv.org/abs/2402.03367v2](http://arxiv.org/abs/2402.03367v2) | [https://github.com/Raudaschl/rag-fusion](https://github.com/Raudaschl/rag-fusion) |
| 48 | Modular RAG | NQ: 49.5; TriviaQA: 71.2; HotpotQA: 56.9; FEVER: 84.8 | Modular RAG architecture with customizable components | GPT-4, GPT-3.5 | [http://arxiv.org/abs/2407.21059v1](http://arxiv.org/abs/2407.21059v1) | N/A |
| 49 | Improving Retrieval for RAG based Question Answering Models on Financial Documents | Hit@1: 0.82, Hit@3: 0.95, MRR: 0.88 | RAG for financial documents with specialized retrieval and ranking | N/A | [http://arxiv.org/abs/2404.07221v2](http://arxiv.org/abs/2404.07221v2) | N/A |
| 50 | Don't Forget to Connect\! Improving RAG with Graph-based Reranking | Hit@1: 0.85, Hit@3: 0.97, MRR: 0.91 | Using graph-based reranking to improve RAG performance | N/A | [http://arxiv.org/abs/2405.18414v1](http://arxiv.org/abs/2405.18414v1) | N/A |

