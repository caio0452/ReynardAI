## 🦊 ReynardAI
A configurable, extensible WIP library for implementing low-latency AI chatbots with retrieval capabilities and/or memory.

**Features:**

* Multi-tiered memory (short-term, long-term, and context compaction)
* Support for any OpenAI-compatible LLM provider
* LLM fallbacks
* Moderation (OpenAI Moderation API, standalone LLM, Llama-guard)
* Vector RAG with easy-to-create knowledge data files in plain `.txt` format
* Built-in vector database (Milvus Lite) that lives in a local database file
* Seamless chatting with multiple users simultaneously
* In-personality rewriting
* Image viewing through a separate VLM

Currently, the library only supports Discord, but it is built to be extensible by implementing a class that inherits from `BaseChatHandler`. Support for more chat platforms will be added in the future.

## 📄 License

This project is licensed under the Apache 2.0 License.
