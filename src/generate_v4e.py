"""Autoregressive V4E generation with compact rhythm buckets."""
from __future__ import annotations
import argparse, json, math
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np
from music21 import note, stream, tempo
from tensorflow import keras
from . import model_v4e  # registers custom objects

ROOT=Path(__file__).resolve().parents[1]; PROCESSED=ROOT/"data"/"processed"; MODELS=ROOT/"models"; OUTPUTS=ROOT/"outputs"

def load(genre):
    arrays=dict(np.load(PROCESSED/f"{genre}_v4e_sequences.npz",allow_pickle=False)); pre=json.loads((PROCESSED/f"{genre}_v4e_metadata.json").read_text()); train=json.loads((MODELS/f"{genre}_training_metadata_v4e.json").read_text()); model=keras.models.load_model(MODELS/f"{genre}_lstm_v4e.keras"); return model,arrays,pre,train

def enc(values,mapping):
    lut={int(k):int(v) for k,v in mapping["value_to_index"].items()}; return np.asarray([lut[int(x)] for x in values.reshape(-1)],np.int32).reshape(values.shape)

def sample(prob,temp,k,rng,forbidden=()):
    p=np.asarray(prob,float).reshape(-1); logits=np.log(np.maximum(p,1e-12))/float(temp); logits[list(forbidden)]=-np.inf; allowed=np.flatnonzero(np.isfinite(logits));
    if k>0 and len(allowed)>k:
        keep=allowed[np.argpartition(logits[allowed],-k)[-k:]]; mask=np.zeros(len(logits),bool); mask[keep]=True; logits[~mask]=-np.inf
    shifted=logits-np.max(logits[np.isfinite(logits)]); q=np.exp(np.where(np.isfinite(shifted),shifted,-np.inf)); q/=q.sum(); return int(rng.choice(len(q),p=q))

def generate(args):
    model,arrays,pre,train=load(args.genre); m=train["mappings"]; reps={"delta":{int(k):int(v) for k,v in pre["representative_decode_steps"]["delta"].items()},"duration":{int(k):int(v) for k,v in pre["representative_decode_steps"]["duration"].items()}}; rng=np.random.default_rng(args.seed); idx=int(rng.integers(len(arrays["X_pitches"]))); seq=int(train["sequence_length"]); slots=int(train["max_pitches_per_onset"]); grid=float(pre["grid_quarter_length"]); eos=int(train["decoder_vocabulary"]["eos"]); start=int(train["decoder_vocabulary"]["start"]); pp=enc(arrays["X_pitches"][idx],m["pitch"]); dd=enc(arrays["X_durations"][idx],m["duration_steps"]); dl=enc(arrays["X_deltas"][idx],m["delta_steps"])
    ins={t.name.split(":")[0]:t for t in model.inputs}; context_model=keras.Model([ins["pitch_input"],ins["duration_input"],ins["delta_input"]],model.get_layer("encoder_dropout").output); emb=model.get_layer("target_pitch_decoder_embedding"); dec=model.get_layer("conditional_pitch_decoder"); ph=model.get_layer("pitch_output"); dh=model.get_layer("duration_output"); deltah=model.get_layer("delta_output"); groups=[]; onset=0.; first_resamples=0
    for _ in range(args.length):
        ctx=context_model([pp[None],dd[None],dl[None]],training=False); dc=sample(deltah(ctx,training=False).numpy()[0],args.delta_temperature,args.top_k,rng); bucket=int(m["delta_steps"]["index_to_value"][dc]); dsteps=reps["delta"][bucket]; onset += dsteps*grid; state=ctx; token=start; pitches=[]; durations=[]; eos_hit=False
        for slot in range(slots):
            out=dec(emb(np.asarray([[token]],np.int32)),initial_state=state,training=False); state=out[:,-1,:]; probs=ph(out,training=False).numpy()[0,0]; pc=sample(probs,args.temperature,args.top_k,rng,{0});
            if pc==eos:
                if slot==0: first_resamples+=1; pc=sample(probs,args.temperature,args.top_k,rng,{0,eos})
                else: eos_hit=True; break
            pitch=int(m["pitch"]["index_to_value"][pc]); df=keras.layers.Concatenate(axis=-1)([out,ph(out,training=False)]); dp=dh(df,training=False).numpy()[0,0]; dclass=sample(dp,args.duration_temperature,args.top_k,rng,{0}); dbucket=int(m["duration_steps"]["index_to_value"][dclass]); pitches.append(pitch); durations.append(reps["duration"][dbucket]); token=pc
        if not pitches: continue
        groups.append({"onset":onset,"delta":dsteps,"pitches":pitches,"durations":durations,"eos":eos_hit,"cap":len(pitches)==slots and not eos_hit})
        npitch=np.zeros(slots,np.int16); ndur=np.zeros(slots,np.int16); npitch[:len(pitches)]=[m["pitch"]["value_to_index"][str(x)] for x in pitches]; ndur[:len(durations)]=[m["duration_steps"]["value_to_index"][str(bucket_for_value(x, reps["duration"]) )] for x in durations]; ndelta=np.asarray([m["delta_steps"]["value_to_index"][str(bucket)]],np.int16)[0]; pp=np.concatenate([pp[1:],npitch[None,:]],0); dd=np.concatenate([dd[1:],ndur[None,:]],0); dl=np.concatenate([dl[1:],np.asarray([ndelta],np.int16)],0)
    out=OUTPUTS/f"{args.genre}_v4e_{args.length}_t{str(args.temperature).replace('.','p')}_k{args.top_k}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mid"; out.parent.mkdir(exist_ok=True); s=stream.Stream(); s.append(tempo.MetronomeMark(number=args.tempo));
    for g in groups:
        for p,d in zip(g["pitches"],g["durations"]): s.insert(g["onset"],note.Note(p,quarterLength=d*grid))
    s.write("midi",fp=out); diagnostics=diagnose(groups)
    print(f"Output MIDI: {out}"); print(f"Seed sequence: {idx} | onset groups: {len(groups)} | notes: {diagnostics['notes']}"); print(json.dumps({"delta_temperature":args.delta_temperature,"duration_temperature":args.duration_temperature,"diagnostics":diagnostics},indent=2)); return out,diagnostics

def bucket_for_value(value,reps):
    return min(reps,key=lambda k:abs(reps[k]-int(value)))

def diagnose(groups):
    ds=[g["delta"] for g in groups]; dur=[d for g in groups for d in g["durations"]]; sizes=[len(g["pitches"]) for g in groups]; pitches=[p for g in groups for p in g["pitches"]]; reps=[max(g["pitches"]) for g in groups]; moves=[abs(reps[i]-reps[i-1]) for i in range(1,len(reps))]; multi=[g["pitches"] for g in groups if len(g["pitches"])>1]; clash=sum(any(abs(a-b)==1 for i,a in enumerate(sorted(set(g))) for b in sorted(set(g))[i+1:]) for g in multi); onset_dist={str(i):round(100*sizes.count(i)/max(1,len(sizes)),3) for i in range(1,6)}; onset_dist["6+"]=round(100*sum(x>=6 for x in sizes)/max(1,len(sizes)),3)
    return {"onset_groups":len(groups),"notes":len(pitches),"delta_bucket_distribution":dict(Counter(ds)),"duration_decoded_distribution":dict(Counter(dur)),"delta_one_percent":100*sum(x==1 for x in ds)/max(1,len(ds)),"duration_one_percent":100*sum(x==1 for x in dur)/max(1,len(dur)),"mean_delta":float(np.mean(ds)) if ds else 0.,"mean_duration":float(np.mean(dur)) if dur else 0.,"total_span":float(max((g["onset"]+max(g["durations"],default=1)*.25 for g in groups),default=0)),"mean_notes_onset":float(np.mean(sizes)) if sizes else 0.,"onset_size_distribution":onset_dist,"eos_rate":100*sum(g["eos"] for g in groups)/max(1,len(groups)),"cap_rate":100*sum(g["cap"] for g in groups)/max(1,len(groups)),"first_eos_resamples":0,"mean_pitch":float(np.mean(pitches)) if pitches else 0.,"pitch_range":[min(pitches),max(pitches)] if pitches else [],"repeated_melody_percent":100*sum(moves[i-1]==0 for i in range(1,len(moves)))/max(1,len(moves)),"over12_percent":100*sum(x>12 for x in moves)/max(1,len(moves)),"one_semitone_clash_percent":100*clash/max(1,len(multi))}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--genre",default="classical"); p.add_argument("--length",type=int,default=200); p.add_argument("--temperature",type=float,default=1.0); p.add_argument("--top-k",type=int,default=10); p.add_argument("--delta-temperature",type=float,default=.8); p.add_argument("--duration-temperature",type=float,default=.8); p.add_argument("--seed",type=int,default=42); p.add_argument("--tempo",type=float,default=100); generate(p.parse_args())
if __name__ == "__main__": main()
