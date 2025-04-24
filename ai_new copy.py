from __future__ import annotations

# ---------------------------------------------------------------------------
# Disable Streamlit’s file-watcher *before* Streamlit is imported
# ---------------------------------------------------------------------------
import os, re
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

from pathlib import Path
from threading import Thread

import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

# ---------------- Model option lists ----------------------------------------
chat_model_options = [
    "braindao/DeepSeek-R1-Distill-Qwen-14B-Blunt-Uncensored-Blunt",
]
image_model_options = [
    "black-forest-labs/FLUX.1-dev",
]
video_model_options = [
    "ByteDance/AnimateDiff-Lightning",
]

# ---------------- Page config ------------------------------------------------
st.set_page_config(page_title="Special JARVIS", page_icon="💬")

# ---------------------------------------------------------------------------
# ♻️ Small utilities ----------------------------------------------------------
# ---------------------------------------------------------------------------

def _fmt(hist):
    """Return the prompt to feed the LLM, adapting to guardrail toggle."""
    if st.session_state.get("guardrails_off", False):
        system = "System: You are an AI assistant. Respond naturally without additional filtering or brevity constraints.\n"
    else:
        system = (
            "System: You are a precise assistant. "
            "Answer in one short sentence unless the user explicitly asks for details.\n"
        )
    prompt = system
    for m in hist:
        prompt += f"{m['role'].capitalize()}: {m['content']}\n"
    return prompt + "Assistant: "


def _needs_continuation(text: str) -> bool:
    return not re.search(r"[\.!?…\u201D\u2019]$", text.strip())


def _dedup_text(text: str) -> str:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if len(paragraphs) >= 2 and paragraphs[-1] == paragraphs[-2]:
        paragraphs.pop()
    return "\n\n".join(paragraphs)


def _derive_title(text: str, max_len: int = 30) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    text = re.split(r"[\.!?]", text, maxsplit=1)[0][:max_len].rstrip()
    return text.capitalize() or "Untitled chat"


def stream_reply(tok, mod, hist, user_msg, *, max_new=256, temp=0.2, top_p=0.95):
    hist.append({"role": "user", "content": user_msg})

    inp = tok(_fmt(hist), return_tensors="pt", truncation=True, max_length=1024).to(mod.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    Thread(target=mod.generate, kwargs=dict(
        **inp,
        streamer=streamer,
        max_new_tokens=max_new,
        temperature=temp,
        top_p=top_p,
        repetition_penalty=1.05,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )).start()

    buf = ""
    for chunk in streamer:
        buf += chunk
        yield buf

    if _needs_continuation(buf):
        extra_ids = mod.generate(
            **inp,
            max_new_tokens=64,
            temperature=temp,
            top_p=top_p,
            repetition_penalty=1.05,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            do_sample=True,
        )[:, inp["input_ids"].shape[1]:]
        continuation = tok.decode(extra_ids[0], skip_special_tokens=True)
        continuation = re.split(r"(?<=[\.!?…])\s", continuation, maxsplit=1)[0]
        buf += continuation
        yield buf

    buf = _dedup_text(buf)
    hist.append({"role": "assistant", "content": buf})

# ---------------------------------------------------------------------------
# 🌐 Sidebar: model pickers & chat management ---------------------------------
# ---------------------------------------------------------------------------

chat_sel = st.sidebar.selectbox("Chat model", chat_model_options)
st.sidebar.markdown("---")

st.sidebar.selectbox("Image model (placeholder)", image_model_options)
st.sidebar.selectbox("Video model (placeholder)", video_model_options)

# 🔒 / 🆓 Guardrail toggle
st.sidebar.markdown("### Safety")
guardrails_off = st.sidebar.checkbox("Disable guardrails (unfiltered)")
st.session_state.guardrails_off = guardrails_off  # store for use in _fmt()

# ---------------- Session state initialisation ------------------------------
if "chats" not in st.session_state:
    st.session_state.chats = {"Chat 1": []}
    st.session_state.current_chat = "Chat 1"

# ➕ Create new chat placeholder
if st.sidebar.button("➕ New chat", use_container_width=True):
    new_title = f"Chat {len(st.session_state.chats) + 1}"
    st.session_state.chats[new_title] = []
    st.session_state.current_chat = new_title

chat_titles = list(st.session_state.chats.keys())
sel_title = st.sidebar.radio("Conversations", chat_titles, index=chat_titles.index(st.session_state.current_chat))
if sel_title != st.session_state.current_chat:
    st.session_state.current_chat = sel_title

# 🗑️ Delete chat (disabled when only one remains)
if st.sidebar.button("🗑️ Delete chat", use_container_width=True, disabled=len(st.session_state.chats) <= 1):
    del_title = st.session_state.current_chat
    st.session_state.chats.pop(del_title, None)
    st.session_state.current_chat = list(st.session_state.chats.keys())[0]
    st.rerun()

HIST = st.session_state.chats[st.session_state.current_chat]

st.sidebar.markdown("---")

# ---------------------------------------------------------------------------
# 💾 Model loading -----------------------------------------------------------
# ---------------------------------------------------------------------------

offload = st.sidebar.checkbox("CPU offload (low VRAM)")
cache_dir = Path(os.getenv("LOCAL_MODEL_ROOT", r"C:/ai/model_cache")).expanduser()
cache_dir.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_chat(model_id: str, offload_to_cpu: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        llm_int8_threshold=6.0,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if offload_to_cpu else device,
        quantization_config=quant_cfg,
        trust_remote_code=True,
        max_memory={0: "7GiB", "cpu": "32GiB"},
    )
    model.eval()
    return tokenizer, model

TOKENIZER, MODEL = st.cache_resource(show_spinner="Loading chat model … (may download once)")(load_chat)(chat_sel, offload)

# ---------------------------------------------------------------------------
# 🔧 Generation controls ------------------------------------------------------
# ---------------------------------------------------------------------------

temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
col1, col2 = st.sidebar.columns(2)
with col1:
    topp = st.slider("Top-p", 0.1, 1.0, 0.95, 0.05)
with col2:
    max_new = st.number_input("Max tokens", 32, 1024, 256, 32)

# ---------------------------------------------------------------------------
# 🖼️ Main panel --------------------------------------------------------------
# ---------------------------------------------------------------------------

st.title("💬 Special JARVIS")

for m in HIST:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Type your message here… (Shift+Enter for new line)")
if prompt:
    # Rename placeholder chat after first user prompt
    if re.fullmatch(r"Chat \d+", st.session_state.current_chat):
        new_title = _derive_title(prompt)
        base = new_title
        suffix = 1
        while new_title in st.session_state.chats:
            suffix += 1
            new_title = f"{base} ({suffix})"
        st.session_state.chats[new_title] = st.session_state.chats.pop(st.session_state.current_chat)
        st.session_state.current_chat = new_title
        st.rerun()

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer_box = st.empty()
        with st.spinner("🧠 Thinking …"):
            part = ""
            for part in stream_reply(
                TOKENIZER,
                MODEL,
                HIST,
                prompt,
                max_new=max_new,
                temp=temp,
                top_p=topp,
            ):
                answer_box.markdown(part + "▌")
        answer_box.markdown(part)

# ---------------------------------------------------------------------------
# 🖼️ Image generation panel  -------------------------------------------------
#   (drop this *after* your current code – it is completely self-contained)
# ---------------------------------------------------------------------------

from diffusers import (
    DiffusionPipeline,
    StableDiffusion3Pipeline,
    SD3Transformer2DModel,
)

# --- UI widgets -------------------------------------------------------------
st.markdown("---")
with st.expander("🖼️  Generate images"):                 # collapsed by default
    img_prompt = st.text_area(
        "Image prompt",
        value="",
        placeholder="Use the same text you just sent to chat (or something new)…",
    )

    # quick-and-dirty inference sliders
    col1, col2 = st.columns(2)
    with col1:
        num_steps = st.slider("Inference steps", 5, 50, 15, 1)
    with col2:
        guidance = st.slider("Guidance scale", 1.0, 20.0, 6.0, 0.5)

    if st.button("🎨 Generate") and img_prompt.strip():
        # --- lazy-load the pipeline (cached after first run) ----------------
        @st.cache_resource(show_spinner="Loading image model … (first time only)")
        def load_image_pipeline(selection: str):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if device == "cuda" else torch.float32

            if selection == "black-forest-labs/FLUX.1-dev":
                pipe = DiffusionPipeline.from_pretrained(
                    selection,
                    torch_dtype=torch_dtype,
                ).to(device)
                pipe.load_lora_weights("Jovie/Midjourney")
            else:
                model_nf4 = SD3Transformer2DModel.from_pretrained(
                    selection,
                    subfolder="transformer",
                    torch_dtype=torch_dtype,
                )
                pipe = StableDiffusion3Pipeline.from_pretrained(
                    selection,
                    cache_dir=cache_dir,          # you already defined this
                    transformer=model_nf4,
                    torch_dtype=torch_dtype,
                ).to(device)

            pipe.enable_model_cpu_offload()
            return pipe

        # use the value chosen in the sidebar’s “Image model” selectbox
        pipe = load_image_pipeline(st.session_state.get("image_model", image_model_options[0]))

        negative_prompt = (
            "low quality, worst quality, blurry, deformed anatomy, watermark, "
            "text, signature, logo, low-res, cropped, mutated hands, mutated face, "
            "extra limbs, extra fingers, poorly drawn eyes, poorly drawn face, "
            "inconsistent style, out of frame"
        )

        # --- run the model ---------------------------------------------------
        with st.spinner("🧑‍🎨  Painting…"):
            images = pipe(
                img_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                max_sequence_length=512,
                num_images_per_prompt=2,
            ).images

        # --- display ---------------------------------------------------------
        for idx, im in enumerate(images, 1):
            st.image(im, caption=f"Result {idx}", use_column_width=True)
