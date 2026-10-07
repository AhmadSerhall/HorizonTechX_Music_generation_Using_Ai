"""Train the experimental compact-rhythm V4E model."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any
import numpy as np
import tensorflow as tf
from tensorflow import keras
from .model_v4e import build_v4e_conditional_model

ROOT = Path(__file__).resolve().parents[1]; PROCESSED = ROOT / "data" / "processed"; MODELS = ROOT / "models"

def _load(genre):
    npz = PROCESSED / f"{genre}_v4e_sequences.npz"; meta = PROCESSED / f"{genre}_v4e_metadata.json"
    if not npz.exists() or not meta.exists(): raise FileNotFoundError(f"Missing V4E artifacts: {npz} / {meta}")
    return dict(np.load(npz, allow_pickle=False)), json.loads(meta.read_text(encoding="utf-8"))

def _encode(values, mapping):
    lut = {int(k): int(v) for k, v in mapping["value_to_index"].items()}
    return np.asarray([lut[int(v)] for v in values.reshape(-1)], dtype=np.int32).reshape(values.shape)

def _prepare(arrays, meta):
    m = meta["mappings"]; pitch = _encode(arrays["X_pitches"], m["pitch"]); dur = _encode(arrays["X_durations"], m["duration_steps"]); delta = _encode(arrays["X_deltas"], m["delta_steps"])
    yp = _encode(arrays["y_pitches"], m["pitch"]); yd = _encode(arrays["y_durations"], m["duration_steps"]); eos = len(m["pitch"]["index_to_value"]); start = eos + 1
    for r in range(len(yp)):
        pads = np.flatnonzero(yp[r] == 0); end = int(pads[0]) if len(pads) else yp.shape[1]
        if end < yp.shape[1]: yp[r, end] = eos; yp[r, end + 1:] = 0; yd[r, end:] = 0
    target = {"pitch_output": yp, "duration_output": yd, "delta_output": _encode(arrays["y_deltas"], m["delta_steps"])}
    decoder = np.zeros_like(yp, dtype=np.int32); decoder[:, 0] = start; decoder[:, 1:] = yp[:, :-1]
    inputs = {"pitch_input": pitch, "duration_input": dur, "delta_input": delta, "target_pitch_input": decoder}
    return inputs, target, eos, start

def _split(arrays):
    ids = np.asarray(arrays["sequence_piece_ids"], dtype=np.int32); pieces = sorted(int(x) for x in np.unique(ids)); cut = min(max(1, round(len(pieces)*.8)), len(pieces)-1); tr, va = pieces[:cut], pieces[cut:]
    return np.flatnonzero(np.isin(ids, tr)), np.flatnonzero(np.isin(ids, va)), tr, va

def _subset(d, idx, maximum):
    if maximum is not None: idx = idx[:maximum]
    return {k: v[idx] for k,v in d.items()}

def _rhythm_metrics(true, probs, labels):
    true = np.asarray(true).reshape(-1); pred = np.asarray(probs).argmax(-1); recalls = {}
    for c in sorted(np.unique(true)):
        mask = true == c; recalls[str(labels[int(c)])] = {"count": int(mask.sum()), "recall_percent": float(np.mean(pred[mask] == c)*100)}
    return {"accuracy_percent": float(np.mean(true == pred)*100), "majority_baseline_percent": float(np.max(np.bincount(true))*100/len(true)), "true_distribution_percent": {str(labels[int(c)]): float(np.mean(true==c)*100) for c in sorted(np.unique(true))}, "predicted_distribution_percent": {str(labels[int(c)]): float(np.mean(pred==c)*100) for c in sorted(np.unique(pred))}, "per_class_recall": recalls}

def _greedy_sizes(model, inputs, eos, start, max_samples=5000):
    sample = {k:v[:max_samples] for k,v in inputs.items() if k != "target_pitch_input"}; tensors={t.name.split(":")[0]:t for t in model.inputs}; ctx_model=keras.Model([tensors["pitch_input"],tensors["duration_input"],tensors["delta_input"]], model.get_layer("encoder_dropout").output); ctx=ctx_model([sample["pitch_input"],sample["duration_input"],sample["delta_input"]],training=False); emb=model.get_layer("target_pitch_decoder_embedding"); dec=model.get_layer("conditional_pitch_decoder"); head=model.get_layer("pitch_output"); token=np.full((len(ctx),),start,np.int32); sizes=np.zeros(len(ctx),np.int32); done=np.zeros(len(ctx),bool)
    for _ in range(8):
        out=dec(emb(token[:,None]),initial_state=ctx,training=False); ctx=out[:,-1,:]; p=head(out,training=False)[:,0,:].numpy().argmax(-1); active=~done; eos_hit=active&(p==eos); sizes[active & ~eos_hit & (p!=0)] += 1; done |= eos_hit | (p==0); token=p
    counts={"zero_notes":int(np.sum(sizes==0)),**{f"{i}_note" if i==1 else f"{i}_notes":int(np.sum(sizes==i)) for i in range(1,6)},"6_plus_notes":int(np.sum(sizes>=6))}; n=max(1,len(sizes)); return {"sample_count":len(sizes),"counts":counts,"percentages":{k:v*100/n for k,v in counts.items()},"mean":float(np.mean(sizes)),"median":float(np.median(sizes)),"maximum":int(np.max(sizes)),"eos_prediction_rate_percent":float(np.mean(done)*100),"cap_hit_rate_percent":float(np.mean((sizes==8)&~done)*100)}

def train(args):
    np.random.seed(args.random_seed); tf.random.set_seed(args.random_seed); arrays, meta = _load(args.genre); inputs, targets, eos, start = _prepare(arrays, meta); tr_idx, va_idx, tr_pieces, va_pieces = _split(arrays); tr_in=_subset(inputs,tr_idx,args.max_sequences); tr_y=_subset(targets,tr_idx,args.max_sequences); va_in=_subset(inputs,va_idx,args.max_sequences); va_y=_subset(targets,va_idx,args.max_sequences); m=meta["mappings"]
    model=build_v4e_conditional_model(meta["sequence_length"],meta["max_pitches_per_onset"],len(m["pitch"]["index_to_value"]),len(m["duration_steps"]["index_to_value"]),len(m["delta_steps"]["index_to_value"]),eos,start); print(model.summary()); print(f"Pieces: {len(tr_pieces)} train / {len(va_pieces)} validation | sequences: {len(tr_in['pitch_input'])}/{len(va_in['pitch_input'])}"); print(f"Classes: pitch={eos-1} real + PAD/EOS, duration={len(m['duration_steps']['index_to_value'])}, delta={len(m['delta_steps']['index_to_value'])} | parameters={model.count_params()}")
    lr=[]
    class Lr(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None): lr.append(float(keras.backend.get_value(self.model.optimizer.learning_rate)))
    hist=model.fit(tr_in,tr_y,validation_data=(va_in,va_y),epochs=args.epochs,batch_size=args.batch_size,callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss",patience=4,restore_best_weights=True),keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=.5,patience=2),Lr()],verbose=2); history={k:[float(x) for x in v] for k,v in hist.history.items()}; best=int(np.argmin(history["val_loss"])+1)
    smoke=args.smoke; MODELS.mkdir(exist_ok=True); model_path=MODELS/(f"{args.genre}_lstm_v4e_smoke.keras" if smoke else f"{args.genre}_lstm_v4e.keras"); meta_path=MODELS/(f"{args.genre}_training_metadata_v4e_smoke.json" if smoke else f"{args.genre}_training_metadata_v4e.json"); model.save(model_path); reloaded=keras.models.load_model(model_path); evals={k:float(v) for k,v in reloaded.evaluate(va_in,va_y,batch_size=args.batch_size,verbose=0,return_dict=True).items()}; preds=reloaded.predict(va_in,batch_size=args.batch_size,verbose=0); dm=_rhythm_metrics(va_y["delta_output"],preds["delta_output"],m["delta_steps"]["index_to_value"]); mask=(va_y["pitch_output"]!=0)&(va_y["pitch_output"]!=eos); durm=_rhythm_metrics(va_y["duration_output"][mask],preds["duration_output"][mask],m["duration_steps"]["index_to_value"]); onset=_greedy_sizes(reloaded,va_in,eos,start)
    result={"version":"V4E_compact_rhythm_EOS","architecture":{"historical_encoder":"pitch Embedding(32) + duration Embedding(16) + delta Embedding(16) -> LSTM(256) -> Dropout(0.3)","decoder":"conditional GRU(256), teacher-forced 8 slots with START/EOS","loss":"unweighted sparse categorical crossentropy; PAD-masked pitch/duration"},"parameter_count":int(model.count_params()),"class_counts":{"pitch_storage":len(m["pitch"]["index_to_value"]),"duration":len(m["duration_steps"]["index_to_value"]),"delta":len(m["delta_steps"]["index_to_value"]),"pitch_decoder":eos+1},"decoder_vocabulary":{"pad":0,"eos":eos,"start":start},"sequence_length":meta["sequence_length"],"max_pitches_per_onset":meta["max_pitches_per_onset"],"mappings":m,"representative_decode_steps":meta["representative_decode_steps"],"train_piece_ids":tr_pieces,"validation_piece_ids":va_pieces,"train_sequence_count":len(tr_in["pitch_input"]),"validation_sequence_count":len(va_in["pitch_input"]),"epochs_completed":len(history["loss"]),"best_epoch":best,"best_validation_loss":float(min(history["val_loss"])),"history":history,"learning_rate_history":lr,"validation_metrics":evals,"delta_metrics":dm,"duration_metrics":durm,"greedy_onset_size":onset,"smoke_test":smoke,"model_path":str(model_path)}; meta_path.write_text(json.dumps(result,indent=2,default=lambda x:x.tolist() if hasattr(x,"tolist") else x),encoding="utf-8"); print(f"Saved model: {model_path}"); print(f"Saved metadata: {meta_path}"); print("Delta metrics:",dm); print("Duration metrics:",durm); print("Greedy onset sizes:",onset); return result

def main():
    p=argparse.ArgumentParser(); p.add_argument("--genre",default="classical"); p.add_argument("--epochs",type=int,default=20); p.add_argument("--batch-size",type=int,default=128); p.add_argument("--max-sequences",type=int); p.add_argument("--smoke",action="store_true"); p.add_argument("--random-seed",type=int,default=42); train(p.parse_args())
if __name__ == "__main__": main()
