# The LLM Pipeline — Study Guide

**AI engineering — study guide**

Four stages, in the order you'd actually explain them in an interview: how the model **learns** from raw text, what's happening **inside** each transformer block, what changes (and doesn't) when you **ask it something**, and how **retrieval** lets it answer using your own documents.

---

## 01 · Training — how the model learns

The goal of training is deceptively simple: given some text, predict the next token. Everything below is in service of doing that one job well, at massive scale.

**Flow:** Internet → Cleaning → Tokenization → Vocabulary → Embeddings → Transformer → Attention → Next-token prediction → Loss → Backpropagation → Updated weights → Trained model

### Preparing the data

**1 — Internet**
A huge scrape of text: web pages, books, code, forums. This raw corpus is the *entire* source of what the model can ever know — there's no other input.

**2 — Cleaning**
Deduplication, quality filtering, and removal of unsafe or low-value content (ads, boilerplate). Quality here caps how good the final model can get.

**3 — Tokenization**
Text is split into sub-word pieces, not whole words — e.g. `unhappiness` → `un` + `happi` + `ness`. This keeps the vocabulary a manageable fixed size while still covering rare words.

**4 — Vocabulary**
The fixed list of every possible token (often 30k–100k+ entries), each with an integer ID. This is the model's entire "alphabet" — it can't represent anything outside it.

### Learning to predict

**5 — Embeddings**
Each token ID is mapped to a dense vector via a lookup table. Tokens with related meaning end up with similar vectors — this is the model's first notion of "meaning."

**6 — Transformer**
A stack of identical blocks (attention + feed-forward) that processes the *whole* input sequence at once, rather than one token at a time. See section 02 for what's inside.

**7 — Attention**
Each token looks at every other token and weighs how relevant it is right now, so context — not just isolated words — shapes the representation.

**8 — Next-token prediction**
After the transformer stack, the model outputs a probability for every token in the vocabulary: "what word comes next?"

### Correcting mistakes

**9 — Loss**
A single number (cross-entropy loss) measuring how wrong the prediction was compared to the actual next token in the training text.

**10 — Backpropagation**
The algorithm that traces the loss backward through every layer, computing exactly how much each weight contributed to the error.

**11 — Updated weights**
An optimizer (e.g. Adam) nudges every weight slightly in the direction that would have reduced the loss.

**12 — Trained model**
Steps 6–11 repeat trillions of times over the dataset. Weights slowly converge into a model that predicts language well.

> **Key idea:** Training = the loop that changes weights. Everything before "trained model" happens once, offline, and costs enormous compute. It never happens again while you're chatting with the model.

---

## 02 · What's inside a transformer block

Step 6 above — "Transformer" — is really a stack of identical blocks repeated N times (32, 96, sometimes more layers). Here's what happens inside one block.

**Flow:** Token → Embedding → Positional encoding → Multi-head attention → Feed-forward network → Residual connections → Output token

**Embedding**
Turns a token into a vector of numbers — its "semantic coordinates" in a high-dimensional space.

**Positional encoding**
Attention alone has no sense of word order, so a position signal is added to each embedding — the model needs to know word 1 came before word 2.

**Multi-head attention**
Several attention computations run in parallel ("heads"), each free to specialize on a different kind of relationship — grammar, coreference, topic — then their outputs are combined. *Attention = "which previous words matter most to me right now?"*

**Feed-forward network**
Once attention has gathered context, a small per-token network processes that information further — applied independently to each position.

**Residual connections**
Each block's input is added back to its output. This preserves the original signal and is a big part of why very deep networks (dozens of layers) are trainable at all.

**Output token**
After N stacked blocks, the final vector is converted back into probabilities over the vocabulary — the model's prediction.

> **Key idea:** The *same* block architecture repeats at every layer. What differs per input is the attention weights — which is exactly what lets one architecture handle wildly different sentences.

---

## 03 · Inference — what happens when you ask it something

This is the same transformer from section 02, run forward only — no learning involved. It's also the part you'll spend the most time working with as an engineer, so it's worth understanding in real depth, not just as a diagram.

**Flow:** User prompt → Tokenize → Embedding → Transformer → Next token → Repeat → Response

### The forward pass

**Tokenize & embed**
Uses the exact same vocabulary and embedding table as training — frozen, no exceptions. Your prompt is just another sequence of token IDs to the model.

**One pass through the transformer**
The prompt runs through all N blocks once. No loss is computed, no backpropagation happens, no weights move — it's pure arithmetic on frozen numbers.

**Output: a probability distribution**
The model doesn't output "a word." It outputs a probability for *every* token in the vocabulary — e.g. 40% chance the next token is "mat," 20% "floor," 0.001% "banana." Something else has to turn that into an actual choice.

> **Worked example — Prompt: "The cat sat on the ___"**
> The model's output layer might produce something like: `mat` 41%, `floor` 18%, `chair` 9%, `sofa` 6%, `roof` 3% … spread across all ~100k tokens in the vocabulary. Decoding (below) decides which of these actually gets picked.

### Choosing the next token (decoding)

**Greedy decoding**
Always pick the single highest-probability token. Simple and deterministic, but can produce dull, repetitive text — it never takes a slightly-lower-probability path even if that leads somewhere better.

**Temperature**
A number that reshapes the probability distribution before picking. Low temperature (e.g. 0.2) sharpens it toward the top choice — more focused, repeatable. High temperature (e.g. 1.2) flattens it — more varied, more surprising, more error-prone.

**Top-k sampling**
Only consider the `k` most likely next tokens (e.g. top 40), discard the long tail, then sample randomly among those. Stops the model from ever picking something absurdly unlikely.

**Top-p (nucleus) sampling**
Instead of a fixed count, keep the smallest set of tokens whose probabilities add up to `p` (e.g. 90%). Adapts automatically — a confident distribution keeps very few tokens, an uncertain one keeps more.

### From one token to a full reply

**Autoregressive loop**
The newly generated token is appended to the input, and the *entire* sequence is run through the model again to get the next one. This is why generation is inherently sequential — token 50 needs tokens 1–49 to already exist.

**Context window**
The maximum number of tokens (prompt + generated reply so far) the model can attend to at once — e.g. 128k tokens. Anything pushed beyond that limit is no longer visible to the model.

**KV cache**
Re-running the full sequence from scratch for every new token would be extremely wasteful. Engines cache each token's attention Key/Value vectors so only the newest token needs fresh computation — this is the main trick that makes chat feel fast.

**Stopping**
Generation ends when the model outputs a special end-of-sequence token, or when a max-token limit set by the caller is hit — whichever comes first.

**Streaming**
Rather than waiting for the full reply, each token is sent to you the moment it's generated — which is why responses appear to "type out" instead of arriving all at once.

**Batching (engineering side)**
Production servers run many users' requests through the model together in one batch to use GPU compute efficiently — a big part of why inference cost and latency depend on traffic patterns, not just model size.

> **Key idea:** No training, no weight updates. The model doesn't "remember" your conversation by learning from it — everything it knows was frozen in at the end of training. What looks like memory within a chat is just the full conversation being re-sent as context on every turn.

---

## 04 · RAG — answering from your own documents

A frozen model can't know about documents that didn't exist during training. Retrieval-augmented generation (RAG) fixes that by fetching relevant text and handing it to the model as context, at inference time.

**Flow:** Documents → Chunking → Embeddings → Vector database → Retriever → Prompt → LLM → Answer

**Chunking**
Documents are split into smaller passages (roughly 300–800 tokens each) so retrieval can be precise and each piece fits comfortably inside the model's context window.

**Embeddings**
Each chunk is converted into a vector, usually with a separate, smaller embedding model — not the LLM itself.

**Vector database**
Stores those chunk vectors, indexed for fast similarity search, with the original text kept as metadata alongside them.

**Retriever**
Embeds the user's question the same way, then finds the nearest chunk vectors by **cosine similarity** — the angle between two vectors; a smaller angle means more related meaning.

**Prompt assembly**
The retrieved chunks are inserted into the prompt next to the question: "here is some context — answer using it."

**Grounded answer**
The LLM generates a normal response, but now grounded in retrieved facts instead of relying only on what it memorized during training.

> **Key idea:** The retriever finds context; the LLM writes the answer. Retrieval is pure similarity search — no understanding involved. All the "reasoning" still happens in the frozen LLM from section 03.

---

## Training vs. inference, side by side

| | Training | Inference |
|---|---|---|
| Forward pass | Yes | Yes |
| Backward pass | Yes | No |
| Loss computed | Yes | No |
| Weights change | Yes | No — frozen |
| Data used | Entire training corpus | Just your prompt + retrieved context |
| Compute cost | Very high, paid once | Lower, paid per request |

---

## Common mix-ups (favorite interview traps)

**Chunk ≠ Token**
A chunk is a passage of text used in RAG (many tokens long). A token is the smallest unit the model itself reads.

**Vector DB stores embeddings**
Not raw PDFs. The original text is kept as metadata, not as the primary searchable object.

**Retriever ≠ reasoner**
The retriever only does similarity search. It doesn't understand the question — the LLM does the actual reasoning once context is retrieved.

**Fine-tuning ≠ RAG**
Fine-tuning further trains the model's weights on a smaller dataset. RAG changes nothing about the model — it only changes what's in the prompt.

**Embedding ≠ token ID**
A token ID is just an index into the vocabulary. An embedding is the dense vector that ID gets mapped to — where meaning actually lives.

---

*Study guide — LLM training, transformer internals, inference, and RAG.*
