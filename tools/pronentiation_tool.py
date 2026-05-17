import torch
import torchaudio
import sys
import gc
import os
from langchain_core.tools import tool
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    AutoModelForCTC
)
import soundfile as sf
import json
import Levenshtein

def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform.squeeze()
RPS_MATRIX = {
    # Vowel Lengthening/Shortening Errors
    "a": ["A"],
    "A": ["a"],
    "i": ["I"],
    "I": ["i"],
    "u": ["U"],
    "U": ["u"],
    "f": ["F"],
    "F": ["f"],
    "x": ["X"],
    "X": ["x"],
    # Aspiration Errors (Deaspiration & Hyper-aspiration)
    "kh": ["k"],
    "gh": ["g"],
    "ch": ["c"],
    "jh": ["j"],
    "Wh": ["W"],
    "qh": ["q"],
    "th": ["t"],
    "dh": ["d"],
    "ph": ["p"],
    "bh": ["b"],
    "k": ["kh"],
    "g": ["gh"],
    "c": ["ch"],
    "j": ["jh"],
    "W": ["Wh"],
    "q": ["qh"],
    "t": ["th"],
    "d": ["dh"],
    "p": ["ph"],
    "b": ["bh"],
    # Place of Articulation Errors (Retroflex to Dental)
    "W": ["t", "Wh"],
    "Wh": ["th", "W"],
    "q": ["d", "qh"],
    "qh": ["dh", "q"],
    "R": ["n"],
    # Sibilant Errors (Palatal/Retroflex to Dental)
    "S": ["s"],
    "z": ["s"],
    # Visarga Errors
    "H": [""]
}

ASPIRATED_CONSONANTS = {'kh', 'gh', 'ch', 'jh', 'Wh', 'qh', 'th', 'dh', 'ph', 'bh'}
UNASPIRATED_CONSONANTS = {'k', 'g', 'c', 'j', 'W', 'q', 't', 'd', 'p', 'b'}
RETROFLEX_CONSONANTS = {'W', 'Wh', 'q', 'qh', 'R', 'z'}
DENTAL_CONSONANTS = {'t', 'th', 'd', 'dh', 'n', 's'}
PALATAL_CONSONANTS = {'c', 'ch', 'j', 'jh', 'Y', 'S'}
SHORT_VOWELS = {'a', 'i', 'u', 'f', 'x'}
LONG_VOWELS = {'A', 'I', 'U', 'F', 'X'}
DIPHTHONGS = {'e', 'o', 'E', 'O'}
SIBILANTS = {'S', 'z', 's'}
# 1. Aspiration Errors (Phonological Feature)
DEASPIRATION_MAP = {
    "kh": "k", "gh": "g", "ch": "c", "jh": "j", "Wh": "W",
    "qh": "q", "th": "t", "dh": "d", "ph": "p", "bh": "b"
}
HYPER_ASPIRATION_MAP = {v: k for k, v in DEASPIRATION_MAP.items()}
# 2. Vowel Length Errors (Phonological Feature)
VOWEL_LENGTHENING_MAP = {"a": "A", "i": "I", "u": "U", "f": "F", "x": "X"}
VOWEL_SHORTENING_MAP = {v: k for k, v in VOWEL_LENGTHENING_MAP.items()}
# 3. Voicing Errors (Phonological Feature)
VOICING_MAP = {
    "k": "g", "c": "j", "W": "q", "t": "d", "p": "b",
    "kh": "gh", "ch": "jh", "Wh": "qh", "th": "dh", "ph": "bh",
}
VOICING_MAP_STOPS = {
    "k": "g", "c": "j", "W": "q", "t": "d", "p": "b",
    "kh": "gh", "ch": "jh", "Wh": "qh", "th": "dh", "ph": "bh"
}
DEVOICING_MAP_STOPS = {v: k for k, v in VOICING_MAP_STOPS.items()}

RETROFLEX_TO_DENTAL_MAP = {
    "W": "t", "Wh": "th", "q": "d", "qh": "dh", "R": "n"
}
DENTAL_TO_RETROFLEX_MAP = {v: k for k, v in RETROFLEX_TO_DENTAL_MAP.items()}
SIBILANT_PALATAL_TO_DENTAL_MAP = {
    "S": "s"
}
SIBILANT_RETROFLEX_TO_DENTAL_MAP = {
    "z": "s"
}
SIBILANT_PALATAL_TO_RETROFLEX_MAP = {
    "S": "z"
}

SIBILANT_RETROFLEX_TO_PALATAL_MAP = {
    "z": "S"
}
R_VOWEL_SIMPLIFICATION_MAP = {
    "f": "i",
    "F": "I"
}
R_VOWEL_OVERCOMPLICATION_MAP = {
    "i": "f",
    "I": "F"
}
ANUSVARA_SUBSTITUTION_MAP = {"M": "n"}


@torch.inference_mode()
def load_model_and_processor(model_path: str):
    """Load tokenizer, feature extractor, processor, and model."""
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_path)
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    model = AutoModelForCTC.from_pretrained(model_path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    model.eval()
    blank_token_id = model.config.pad_token_id
    return processor, model, blank_token_id

@torch.inference_mode()
def get_model_prediction(processor, model, input_audio: torch.Tensor):
    """Run model inference and decode phonemes."""
    input_values = processor(input_audio, return_tensors="pt", sampling_rate=16000).input_values
    logits = model(input_values).logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription_str = processor.batch_decode(predicted_ids)[0]
    predicted_phonemes = list(transcription_str.replace(" ", "|"))
    return {
        "predicted_phonemes": predicted_phonemes,
        "log_probs": log_probs,
        "logits": logits,
        "transcription": transcription_str
    }

def calculate_gop_score(log_probs, phoneme_ids, target_idx, blank_token_id):
    """Compute GOP (Goodness of Pronunciation) acoustic likelihood difference."""
    try:
        log_probs_for_loss = log_probs.permute(1, 0, 2)
        input_lengths = torch.tensor([log_probs_for_loss.shape[0]])
        ctc_loss_fn = torch.nn.CTCLoss(blank=blank_token_id, reduction='sum', zero_infinity=True)
        targets = torch.tensor(phoneme_ids, dtype=torch.long)
        target_lengths = torch.tensor([len(phoneme_ids)])
        nll_canonical = ctc_loss_fn(log_probs_for_loss, targets, input_lengths, target_lengths)
        modified_ids = phoneme_ids[:target_idx] + phoneme_ids[target_idx + 1:]
        if modified_ids:
            modified_targets = torch.tensor(modified_ids, dtype=torch.long)
            modified_lengths = torch.tensor([len(modified_ids)])
            nll_modified = ctc_loss_fn(log_probs_for_loss, modified_targets, input_lengths, modified_lengths)
        else:
            nll_modified = torch.tensor(0.0)

        return (nll_modified - nll_canonical).item()
    except Exception:
        return float("nan")


def calculate_gop_margin(processor, logits: torch.Tensor, target_phoneme: str):
    """Compute GOP margin between target phoneme and strongest competitor."""
    rps_set = RPS_MATRIX.get(target_phoneme, [])
    if not rps_set:
        return float("inf"), "N/A"

    try:
        target_id = processor.tokenizer.convert_tokens_to_ids(target_phoneme)
        competitor_ids = processor.tokenizer.convert_tokens_to_ids(rps_set)
        max_target = torch.max(logits[:, :, target_id]).item()

        max_competitor = -float("inf")
        strongest_comp = "N/A"
        for i, comp_id in enumerate(competitor_ids):
            max_comp = torch.max(logits[:, :, comp_id]).item()
            if max_comp > max_competitor:
                max_competitor = max_comp
                strongest_comp = rps_set[i]

        return max_target - max_competitor, strongest_comp
    except Exception:
        return float("nan"), "N/A"


def calculate_gop_var_logit(processor, logits: torch.Tensor, target_phoneme: str):
    """Compute variance of logit activations for a phoneme."""
    try:
        target_id = processor.tokenizer.convert_tokens_to_ids(target_phoneme)
        return torch.var(logits[0, :, target_id]).item()
    except Exception:
        return float("nan")

def generate_overall_phoneme_scores(processor, model, blank_token_id, loaded_audio: torch.Tensor, canonical_transcript: str):
    """Compute GOP, margin, and variance scores for all phonemes in the input."""
    pred = get_model_prediction(processor, model, loaded_audio)
    logits, log_probs = pred['logits'], pred['log_probs']
    phonemes = list(canonical_transcript.replace(" ", "|"))
    phoneme_ids = processor.tokenizer.convert_tokens_to_ids(phonemes)

    results = []
    for i, p in enumerate(phonemes):
        gop_af = calculate_gop_score(log_probs, phoneme_ids, i, blank_token_id)
        gop_margin, competitor = calculate_gop_margin(processor, logits, p)
        var_logit = calculate_gop_var_logit(processor, logits, p)
        results.append({
            "Phoneme": p,
            "GOP_AF": gop_af,
            "GOP_Margin": gop_margin,
            "Strongest_Competitor": competitor,
            "Var_Logit": var_logit
        })
    return results

def get_phonetic_analysis(processor, model, loaded_audio: torch.Tensor, ground_truth_text: str):
    """End-to-end phonetic analysis pipeline."""
    gt_phonemes = list(ground_truth_text.replace(" ", "|"))
    prediction = get_model_prediction(processor, model, loaded_audio)
    return {
        "transcription": prediction['transcription'],
        "ground_truth_phonemes": gt_phonemes,
        "predicted_phonemes": prediction['predicted_phonemes'],
        "ground_truth_text": ground_truth_text
    }


def get_phoneme_segment_details(phoneme_score_vectors: list, index: int) -> dict:
    """
    Retrieves the detailed score vector for a phoneme segment at a given index.
    In a real system, this would be more complex, mapping ASR output segments
    to their full scores. For this exercise, we assume phoneme_score_vectors
    is already aligned with the ground truth for correct/replace operations,
    and separate entries for insertions.
    """
    if 0 <= index < len(phoneme_score_vectors):
        return phoneme_score_vectors[index]
    return {
        'Phoneme': 'UNKNOWN',
        'GOP_AF': -999.0,
        'GOP_Margin': -999.0,
        'Strongest_Competitor': 'N/A'
    }


def run_substitution_rule_engine(score_vector: dict) -> dict:
    MARGIN_THRESHOLD = 0.3
    target_phoneme = score_vector.get('Phoneme')
    gop_margin = score_vector.get('GOP_Margin')
    strongest_competitor = score_vector.get('Strongest_Competitor')

    # --- 1. Aspiration Errors ---
    if target_phoneme in ASPIRATED_CONSONANTS and strongest_competitor == DEASPIRATION_MAP.get(target_phoneme):
        return {'type': 'De-aspiration', 'details': {'confused_with': strongest_competitor}}
    elif target_phoneme in UNASPIRATED_CONSONANTS and strongest_competitor == HYPER_ASPIRATION_MAP.get(target_phoneme):
        return {'type': 'Hyper-aspiration', 'details': {'confused_with': strongest_competitor}}

    # --- 2. Vowel Length Errors ---
    elif target_phoneme in LONG_VOWELS and strongest_competitor == VOWEL_SHORTENING_MAP.get(target_phoneme):
        return {'type': 'Vowel Shortening', 'details': {'confused_with': strongest_competitor}}
    elif target_phoneme in SHORT_VOWELS and strongest_competitor == VOWEL_LENGTHENING_MAP.get(target_phoneme):
        return {'type': 'Vowel Lengthening', 'details': {'confused_with': strongest_competitor}}

    # --- 3. Voicing Errors (Stops only) ---
    elif target_phoneme in VOICING_MAP_STOPS and strongest_competitor == VOICING_MAP_STOPS.get(target_phoneme):
        return {'type': 'Voicing Error (Unvoiced to Voiced)', 'details': {'confused_with': strongest_competitor}}
    elif target_phoneme in DEVOICING_MAP_STOPS and strongest_competitor == DEVOICING_MAP_STOPS.get(target_phoneme):
        return {'type': 'De-voicing Error (Voiced to Unvoiced)', 'details': {'confused_with': strongest_competitor}}

    # --- 4. Place of Articulation Errors - Specific Consonants (Stops/Nasals) ---
    elif target_phoneme in RETROFLEX_TO_DENTAL_MAP and strongest_competitor == RETROFLEX_TO_DENTAL_MAP.get(target_phoneme):
        return {'type': 'Place Shift: Retroflex to Dental', 'details': {'confused_with': strongest_competitor}}
    elif target_phoneme in DENTAL_TO_RETROFLEX_MAP and strongest_competitor == DENTAL_TO_RETROFLEX_MAP.get(target_phoneme):
        return {'type': 'Place Shift: Dental to Retroflex', 'details': {'confused_with': strongest_competitor}}

    # --- 5. Place of Articulation Errors - Sibilants ---
    elif target_phoneme in SIBILANT_PALATAL_TO_DENTAL_MAP and strongest_competitor == SIBILANT_PALATAL_TO_DENTAL_MAP.get(target_phoneme):
        return {'type': 'Sibilant Place Shift: Palatal to Dental', 'details': {'confused_with': strongest_competitor}}
    elif target_phoneme in SIBILANT_RETROFLEX_TO_DENTAL_MAP and strongest_competitor == SIBILANT_RETROFLEX_TO_DENTAL_MAP.get(target_phoneme):
        return {'type': 'Sibilant Place Shift: Retroflex to Dental', 'details': {'confused_with': strongest_competitor}}
    elif target_phoneme in SIBILANT_PALATAL_TO_RETROFLEX_MAP and strongest_competitor == SIBILANT_PALATAL_TO_RETROFLEX_MAP.get(target_phoneme):
        return {'type': 'Sibilant Place Shift: Palatal to Retroflex', 'details': {'confused_with': strongest_competitor}}
    elif target_phoneme in SIBILANT_RETROFLEX_TO_PALATAL_MAP and strongest_competitor == SIBILANT_RETROFLEX_TO_PALATAL_MAP.get(target_phoneme):
        return {'type': 'Sibilant Place Shift: Retroflex to Palatal', 'details': {'confused_with': strongest_competitor}}

    # --- 6. R-Vowel Simplification (Vowel Quality) ---
    elif target_phoneme in R_VOWEL_SIMPLIFICATION_MAP and strongest_competitor == R_VOWEL_SIMPLIFICATION_MAP.get(target_phoneme):
        return {'type': 'R-Vowel Simplification', 'details': {'confused_with': strongest_competitor}}
    elif target_phoneme in R_VOWEL_OVERCOMPLICATION_MAP and strongest_competitor == R_VOWEL_OVERCOMPLICATION_MAP.get(target_phoneme):
        return {'type': 'R-Vowel Overcomplication', 'details': {'confused_with': strongest_competitor}}

    elif target_phoneme == "M" and strongest_competitor == ANUSVARA_SUBSTITUTION_MAP.get("M"): # Check if 'M' was replaced by 'n'
        return {'type': 'Anusvara Simplification', 'details': {'confused_with': strongest_competitor}}

    elif strongest_competitor != 'N/A' and gop_margin is not None and gop_margin < MARGIN_THRESHOLD:
        return {
            'type': 'Imprecise Pronunciation',
            'details': {
                'reason': 'The phoneme was weakly pronounced and almost confused with another sound.',
                'strongest_competitor': strongest_competitor,
                'gop_margin': gop_margin
            }
        }
    return {
        'type': 'Abnormal Substitution Error',
        'details': {'strongest_competitor': strongest_competitor}
    }


def diagnose_errors_classification(phoneme_score_vectors: list, result: dict) -> list:
    """
    Classifies each phoneme and runs the rule engine for incorrect ones.
    This corrected version uses Levenshtein.opcodes to handle all
    operations (equal, replace, insert, delete) and works with
    lists of phonemes to support multi-character phonemes.
    """
    ground_truth_phonemes = result["ground_truth_phonemes"]
    predicted_phonemes = result["predicted_phonemes"]

    diagnoses_gt_aligned = []
    for i, gt_ph in enumerate(ground_truth_phonemes):
        vector = get_phoneme_segment_details(phoneme_score_vectors, i)
        diagnoses_gt_aligned.append({
            'phoneme': gt_ph,
            'classification': "CORRECT", # Assume correct
            'error_type': None,
            'details': {},
            'gop_score': vector['GOP_AF'],
            'gop_margin': vector['GOP_Margin'],
            'strongest_competitor': vector['Strongest_Competitor']
        })

    opcodes = Levenshtein.opcodes(predicted_phonemes, ground_truth_phonemes)

    final_diagnoses = []
    for op_type, pred_start, pred_end, gt_start, gt_end in opcodes:

        if op_type == "equal":
            for i in range(gt_start, gt_end):
                final_diagnoses.append(diagnoses_gt_aligned[i])
        elif op_type == "replace":
            # 🚨 THE FIX: Extract exactly what the model heard from Levenshtein
            for i, gt_idx in enumerate(range(gt_start, gt_end)):
                vector = diagnoses_gt_aligned[gt_idx]
                substitution_error = run_substitution_rule_engine(vector)

                # Grab the actual phoneme the model heard instead of 'None'
                pred_idx = pred_start + i
                if pred_idx < len(predicted_phonemes):
                    actual_heard = predicted_phonemes[pred_idx]
                    substitution_error['details']['strongest_competitor'] = actual_heard

                vector.update({
                        "classification": "INCORRECT",
                        "error_type": substitution_error['type'],
                        "details": substitution_error['details']
                    })
                final_diagnoses.append(vector)


        elif op_type == 'insert':
            for i in range(pred_start, pred_end):
                inserted_phoneme = predicted_phonemes[i]
                if inserted_phoneme == '|':
                    final_diagnoses.append({
                        "phoneme": "|",
                        "classification": "INCORRECT",
                        "error_type": "Unnatural Pause/Space",
                        "details": {"reason": "An unscripted pause was detected."}
                    })
                else:
                    final_diagnoses.append({
                        "phoneme": inserted_phoneme,
                        "classification": "INCORRECT",
                        "error_type": "Insertion",
                        "details": {"inserted_phoneme": inserted_phoneme}
                    })

        elif op_type == 'delete':
            for i in range(gt_start, gt_end):
                omitted_phoneme = ground_truth_phonemes[i]
                # Ground truth phoneme was not produced by the model
                final_diagnoses.append({
                    "phoneme": omitted_phoneme,
                    "classification": "INCORRECT",
                    "error_type": "Deletion",
                    "details": {"missing_phoneme": omitted_phoneme}
                })
                # Also update the aligned diagnosis vector if available
                if i < len(diagnoses_gt_aligned):
                    vector = diagnoses_gt_aligned[i]
                    if omitted_phoneme == 'H':
                        etype = 'Visarga Deletion'
                    else:
                        etype = 'Omission'
                    vector.update({
                        "classification": "INCORRECT",
                        "error_type": etype,
                        "details": {"missing_phoneme": omitted_phoneme}
                    })
                    final_diagnoses.append(vector)
    for diag in final_diagnoses:
        diag.pop('gop_score', None)
        diag.pop('gop_margin', None)
        diag.pop('strongest_competitor', None)

    return final_diagnoses



sys.path.insert(0, os.path.abspath(".."))
MODEL_PATH = r"artifacts\xlsr-53-saved-model"
processor = None
mdd_model = None
blank_token_id = None

print("🧘‍♂️ Guru's Acoustic Senses Awakening (Loading Wav2Vec2 Model)... ")

try:
    processor, mdd_model, blank_token_id = load_model_and_processor(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdd_model.to(device)
    print("✅ Acoustic Senses Fully Loaded!")
except Exception as e:
    print(f"⚠️ Warning: Could not load MDD model. Make sure the MODEL_PATH is correct. Error: {e}")



@tool
def speech_to_phoneme_slp1(audio_file_path: str) -> str:
    """
    Step 1 of Pronunciation Analysis.
    Use this tool FIRST when a user uploads an audio file to practice pronunciation.
    Input: The absolute path to the user's .wav audio file.
    Returns: The predicted phonemes transcribed strictly in SLP1 encoding.
    """
    if processor is None or mdd_model is None:
        return "Error: Acoustic model not loaded."

    try:
        loaded_audio = load_audio(audio_file_path)
        prediction = get_model_prediction(processor, mdd_model, loaded_audio)
        predicted_slp1 = prediction.get('transcription', '').strip()

        return f"Predicted SLP1 Phonemes: {predicted_slp1}"
    except Exception as e:
        return f"ASR Error: {str(e)}"
