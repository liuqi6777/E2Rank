"""Screen ReasonRank against BRIGHT; write local audit outputs under DATA_DIR/results."""
import argparse
from pathlib import Path
import re
import json
import math
from collections import Counter, defaultdict

from prepare_reasonrank import (
    ROOT, normalize as norm, query_key as loose, document_key,
    parse_query_and_passages, parse_teacher_ranking, _user_prompt,
)


def stats(values):
    import numpy as np
    return {k: round(float(v), 3) for k, v in zip(
        ['min', 'p25', 'median', 'p75', 'p95', 'max'],
        np.percentile(values, [0, 25, 50, 75, 95, 100]))} if values else {}


def ndcg(final, relevant):
    relevant = set(relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(relevant))))
    return sum((d in relevant) / math.log2(i + 2)
               for i, d in enumerate(final[:10])) / ideal if ideal else 0


def audit(data_dir):
    import numpy as np
    import pyarrow.parquet as pq
    from sklearn.feature_extraction.text import TfidfVectorizer
    from rapidfuzz.fuzz import ratio, token_set_ratio

    BASE = Path(data_dir)
    OUT = BASE / 'results'
    if not list((BASE / 'bright/examples').glob('*.parquet')):
        raise FileNotFoundError('No BRIGHT examples; run download_reasonrank_audit.py first')
    OUT.mkdir(parents=True, exist_ok=True)
    rr=[];errors=[]
    for split in ['train','val']:
     for idx,row in enumerate(pq.read_table(BASE/'reasonrank'/f'{split}.parquet').to_pylist()):
      try:
       query,docs=parse_query_and_passages(_user_prompt(row['prompt']))
       ids=row['initial_list'];ranking=parse_teacher_ranking(row['label'],len(ids))
       issues=[]
       if len(ids)!=len(docs):issues.append('passage_count_mismatch')
       if [ids[i-1] for i in ranking]!=row['final_list']:issues.append('final_list_mismatch')
       if len(set(ids))!=len(ids):issues.append('duplicate_document_ids')
       if not set(row['relevant_docids']).issubset(ids):issues.append('relevant_ids_outside_list')
       if row['reward_model']['ground_truth']!=row['label']:issues.append('reward_label_mismatch')
       for k in ['initial_list','final_list','relevant_docids']:
        if row['extra_info'][k]!=row[k]:issues.append('extra_info_'+k+'_mismatch')
       if issues:errors.append({'split':split,'row':idx,'issues':issues})
       rel=set(row['relevant_docids']);keep=ids[:16]
       rr.append(dict(split=split,row=idx,source=row['dataset'],query=query,norm=norm(query),loose=loose(query),ids=ids,docs=docs,ranking=ranking,relevant=list(rel),n=len(ids),p=len(rel),ndcg=ndcg(row['final_list'],rel),top1_relevant=row['final_list'][0] in rel,keep16_p=len(rel.intersection(keep)),teacher_top1_kept=ranking[0]<=16,doc_hashes=[document_key(d) for d in docs]))
      except Exception as e:errors.append({'split':split,'row':idx,'error':str(e)})
    bright=[]
    for p in sorted((BASE/'bright/examples').glob('*.parquet')):
     domain=p.name.split('-00000')[0]
     for row in pq.read_table(p).to_pylist():bright.append(dict(domain=domain,id=row['id'],query=row['query'],norm=norm(row['query']),loose=loose(row['query']),gold=row['gold_ids']))
    summary={'input_rows':{split:pq.ParquetFile(BASE/'reasonrank'/f'{split}.parquet').metadata.num_rows for split in ['train','val']},'parse_and_structure_errors':errors,'bright_counts':dict(Counter(b['domain'] for b in bright)),'splits':{}}
    for split in ['train','val']:
     rows=[r for r in rr if r['split']==split];by_source={}
     for source in sorted(set(r['source'] for r in rows)):
      a=[r for r in rows if r['source']==source];ok=[r for r in a if r['n']>=16]
      by_source[source]={'rows':len(a),'unique_queries':len(set(r['norm'] for r in a)),'candidates':sum(r['n'] for r in a),'positive_counts':dict(Counter(r['p'] for r in a)),'teacher_ndcg_mean':sum(r['ndcg'] for r in a)/len(a),'teacher_ndcg_lt_04':sum(r['ndcg']<.4 for r in a),'teacher_top1_not_relevant':sum(not r['top1_relevant'] for r in a),'retained16_rows':len(ok),'retained16_loses_all_positives':sum(r['p']>0 and r['keep16_p']==0 for r in ok)}
     ok=[r for r in rows if r['n']>=16]
     summary['splits'][split]={'rows':len(rows),'sources':by_source,'candidate_count_histogram':dict(Counter(r['n'] for r in rows)),'unique_queries':len(set(r['norm'] for r in rows)),'unique_documents_by_text':len(set(h for r in rows for h in r['doc_hashes'])),'candidate_occurrences':sum(r['n'] for r in rows),'positive_occurrences':sum(r['p'] for r in rows),'no_positive':sum(r['p']==0 for r in rows),'all_positive':sum(r['p']==r['n'] for r in rows),'duplicate_text_within_list':sum(len(set(r['doc_hashes']))<r['n'] for r in rows),'query_words':stats([len(r['query'].split()) for r in rows]),'document_words':stats([len(d.split()) for r in rows for d in r['docs']]),'teacher_ndcg':stats([r['ndcg'] for r in rows]),'teacher_ndcg_lt_04':sum(r['ndcg']<.4 for r in rows),'teacher_top1_not_relevant':sum(not r['top1_relevant'] for r in rows),'retained16_rows':len(ok),'retained16_dropped_rows':len(rows)-len(ok),'retained16_positive_occurrences_before':sum(r['p'] for r in ok),'retained16_positive_occurrences_after':sum(r['keep16_p'] for r in ok),'retained16_loses_all_positives':sum(r['p']>0 and r['keep16_p']==0 for r in ok),'retained16_loses_teacher_top1':sum(not r['teacher_top1_kept'] for r in ok)}
    qgroups=defaultdict(list)
    for i,r in enumerate(rr):qgroups[r['norm']].append(i)
    duplicates=[v for v in qgroups.values() if len(v)>1]
    summary['duplicate_query_groups']=[{'rows':[(rr[i]['split'],rr[i]['row'],rr[i]['source']) for i in v],'query':rr[v[0]]['query']} for v in duplicates]
    summary['train_val_exact_overlap_queries']=sum(len({rr[i]['split'] for i in v})>1 for v in duplicates)
    # Candidate document overlap is a separate quantity from query leakage.
    trdoc={h for r in rr if r['split']=='train' for h in r['doc_hashes']};vadoc={h for r in rr if r['split']=='val' for h in r['doc_hashes']}
    summary['train_val_shared_document_texts']=len(trdoc & vadoc)
    print('Structural audit complete',flush=True)
    texts=[r['query'] for r in rr]+[b['query'] for b in bright]
    vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,max_features=120000,sublinear_tf=True,dtype=np.float32)
    X=vec.fit_transform(texts);R=X[:len(rr)];B=X[len(rr):]
    print('Similarity representation ready',X.shape,flush=True)
    hits=[]
    for start in range(0,len(bright),64):
     sims=(B[start:start+64]@R.T).toarray()
     for off,scores in enumerate(sims):
      b=bright[start+off]
      inds=np.argsort(scores)[-5:][::-1]
      for rank,i in enumerate(inds):
       r=rr[int(i)];sim=float(scores[i]);exact=b['norm']==r['norm'];lexact=b['loose']==r['loose']
       if rank==0 or sim>=.70 or exact or lexact:
        hits.append({'bright_domain':b['domain'],'bright_id':b['id'],'rr_split':r['split'],'rr_row':r['row'],'rr_source':r['source'],'cosine':round(sim,6),'exact':exact,'loose_exact':lexact,'edit_ratio':round(ratio(b['norm'],r['norm']),3),'token_set_ratio':round(token_set_ratio(b['loose'],r['loose']),3),'bright_query':b['query'],'rr_query':r['query']})
    (OUT/'bright_query_matches.json').write_text(json.dumps(hits,ensure_ascii=False,indent=2))
    summary['bright_overlap']={}
    for name,pred in [('exact',lambda h:h['exact']),('loose_exact',lambda h:h['loose_exact']),('cosine_ge_090',lambda h:h['cosine']>=.9),('cosine_ge_080',lambda h:h['cosine']>=.8),('cosine_ge_070',lambda h:h['cosine']>=.7)]:
     a=[h for h in hits if pred(h)];summary['bright_overlap'][name]={'bright_queries':len(set((h['bright_domain'],h['bright_id']) for h in a)),'reasonrank_rows':len(set((h['rr_split'],h['rr_row']) for h in a)),'bright_by_domain':dict(Counter(d for d,_ in set((h['bright_domain'],h['bright_id']) for h in a))),'rr_by_split':dict(Counter(sp for sp,_ in set((h['rr_split'],h['rr_row']) for h in a)))}
    print('BRIGHT similarity complete',summary['bright_overlap'],flush=True)
    # High-similarity internal query pairs, including train/validation overlap.
    internal=[]
    for start in range(0,len(rr),96):
     sims=(R[start:start+96]@R.T).toarray()
     for off,scores in enumerate(sims):
      i=start+off
      for j in np.flatnonzero(scores>=.90):
       if j<=i:continue
       a,b=rr[i],rr[int(j)]
       internal.append({'a_split':a['split'],'a_row':a['row'],'a_source':a['source'],'b_split':b['split'],'b_row':b['row'],'b_source':b['source'],'cosine':round(float(scores[j]),6),'exact':a['norm']==b['norm'],'a_query':a['query'],'b_query':b['query']})
    (OUT/'internal_query_matches.json').write_text(json.dumps(internal,ensure_ascii=False,indent=2))
    summary['internal_similarity_ge_090']={'pairs':len(internal),'cross_split_pairs':sum(h['a_split']!=h['b_split'] for h in internal)}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    # Preserve lightweight row metadata for an auditable exclusion manifest.
    (OUT/'record_metadata.json').write_text(json.dumps([{k:v for k,v in r.items() if k not in ['docs','doc_hashes','norm','loose']} for r in rr],ensure_ascii=False))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

    # Independent containment screen catches title removal and truncated questions.
    containment=[]
    for b in bright:
     for r in rr:
      short,long=sorted([b['loose'],r['loose']],key=len)
      if len(short)>=80 and len(short.split())>=12 and (' '+short+' ') in (' '+long+' '):
       containment.append({'bright_domain':b['domain'],'bright_id':b['id'],'rr_split':r['split'],'rr_row':r['row'],'rr_source':r['source'],'bright_query':b['query'],'rr_query':r['query']})
    (OUT/'containment_matches.json').write_text(json.dumps(containment,ensure_ascii=False,indent=2))
    quarantine={(h['rr_split'],h['rr_row']) for h in hits if h['cosine']>=.7 or h['loose_exact']}
    new_containment=[h for h in containment if (h['rr_split'],h['rr_row']) not in quarantine]
    quarantine.update((h['rr_split'],h['rr_row']) for h in containment)
    summary['additional_containment_matches']=len(new_containment)
    summary['conservative_quarantine']={'rule':'char cosine >= 0.70 OR punctuation-normalized exact OR >=80-character/12-word whole-query containment; screening, NOT confirmed duplicate count','rows':len(quarantine),'train_rows_remaining':sum(r['split']=='train' and (r['split'],r['row']) not in quarantine for r in rr),'sources_remaining':dict(Counter(r['source'] for r in rr if r['split']=='train' and (r['split'],r['row']) not in quarantine))}
    conflicts=[]
    for r in rr:
     labels=defaultdict(set)
     for id_,h in zip(r['ids'],r['doc_hashes']):labels[h].add(id_ in r['relevant'])
     if any(len(v)>1 for v in labels.values()):conflicts.append({'split':r['split'],'row':r['row'],'source':r['source']})
    summary['identical_text_conflicting_relevance']={'rows':len(conflicts),'by_split':dict(Counter(r['split'] for r in conflicts))}
    loosegroups=defaultdict(list)
    for r in rr:loosegroups[r['loose']].append((r['split'],r['row']))
    summary['internal_loose_duplicate_groups']=[v for v in loosegroups.values() if len(v)>1]
    (OUT/'quarantine_manifest.json').write_text(json.dumps({'note':summary['conservative_quarantine']['rule'],'rows':[{'split':sp,'row':idx} for sp,idx in sorted(quarantine)]},indent=2))
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps({k:summary[k] for k in ['additional_containment_matches','conservative_quarantine','identical_text_conflicting_relevance','internal_loose_duplicate_groups']},indent=2),flush=True)
    print('New containment matches:',json.dumps(new_containment,ensure_ascii=False),flush=True)
    (OUT/'conflicting_relevance_manifest.json').write_text(json.dumps(conflicts,indent=2))
    excluded=quarantine | {(r['split'],r['row']) for r in conflicts}
    # Conservative usable-count estimate: binary reward needs both label classes;
    # identical query groups keep one representative. This is not a finalized split.
    seen=set();usable=[]
    for r in rr:
     if r['split']!='train' or (r['split'],r['row']) in excluded or not 0<r['p']<r['n'] or r['loose'] in seen:continue
     seen.add(r['loose']);usable.append(r)
    summary['conservative_usable_estimate']={'rows':len(usable),'reason_only_rows':sum(r['source']!='msmarco' for r in usable),'sources':dict(Counter(r['source'] for r in usable)),'candidate_occurrences':sum(r['n'] for r in usable),'positive_occurrences':sum(r['p'] for r in usable)}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print('Conservative usable estimate',summary['conservative_usable_estimate'],flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/audit_reasonrank_bright')
    args = parser.parse_args()
    audit(args.data_dir)


if __name__ == '__main__':
    main()
