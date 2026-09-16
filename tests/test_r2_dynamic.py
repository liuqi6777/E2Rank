"""Dynamic R2 executes real retrieval rewards with isolated action randomness."""
from dataclasses import fields
import json
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3Model

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
import run_g1_r2_dynamic as run
from train_g1_r2_dynamic import DynamicRLArguments, SeededDynamicModel
from grpo_trainer import rollout_rng_contract


def test_dynamic_configuration_and_frozen_evaluation():
    rows, _, directory = run.resolve()
    assert len(rows)==3
    for row in rows:
        cfg=row['config']
        args=DynamicRLArguments(**{f.name:cfg[f.name] for f in fields(DynamicRLArguments) if f.name in cfg})
        assert args.rollout_seed==cfg['seed']==cfg['data_seed']
        assert args.action_components==(('query',),)
        assert args.reward_terms[0].k==10
        assert args.dynamic_retrieval_k==20 and args.group_size==64
        command=run.evaluation_command(row,directory)
        assert command[command.index('--model')+1]==cfg['output_dir']
        assert command[command.index('--fixed_corpus_model')+1]==cfg['model_name_or_path']
        assert command[command.index('--fixed_corpus_model_revision')+1]==run.night.REVISION
        fixed=json.loads(command[command.index('--fixed_corpus_model_kwargs')+1])
        assert fixed['max_length']==8192
    with pytest.raises(ValueError):
        DynamicRLArguments(dynamic_retrieval=True,rollout_seed=None)
    with pytest.raises(ValueError):
        DynamicRLArguments(dynamic_retrieval=True,rollout_seed=42,gradient_estimator='conditional_projection')


def test_dynamic_rng_and_real_full_corpus_gradient():
    previous=torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(5)
        backbone=Qwen3Model(Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,head_dim=8,
            max_position_embeddings=32,attention_dropout=0.,use_cache=False))
        class Index:
            def __init__(self):
                self.vectors=F.normalize(torch.randn(32,16),dim=-1)
                self.calls=[]
            def search(self,actions,k,route_ids=None):
                self.calls.append((actions.detach().clone(),k))
                return (actions@self.vectors.T).topk(k,dim=-1)
        index=Index()
        cfg=run.resolve(seeds=[42])[0][0]['config']
        args=DynamicRLArguments(**{f.name:cfg[f.name] for f in fields(DynamicRLArguments) if f.name in cfg})
        model=SeededDynamicModel(backbone,index,args)
        model.train()
        model.policy.exploration.set_step(0,113)
        means=F.normalize(torch.randn(2,16),dim=-1)
        rng=torch.random.get_rng_state().clone()
        a=model.policy.sample(means)
        assert torch.equal(rng,torch.random.get_rng_state())
        model.policy.rollout_rng.reset(42)
        torch.randn(100)
        b=model.policy.sample(means)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        assert rollout_rng_contract(model.policy)['seed']==42
        ids=torch.randint(1,32,(2,5))
        out=model(query=dict(input_ids=ids,attention_mask=torch.ones_like(ids)),
            candidate_ordinals=torch.tensor([[0,1,2],[3,4,5]]),
            relevance_labels=torch.tensor([[3.,2.,1.],[3.,2.,1.]]))
        assert index.calls[-1][0].shape==(2,64,16) and index.calls[-1][1]==20
        assert torch.isfinite(out.loss) and 0 <= out.reward <= 1
        out.loss.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.double().square().sum().item() for g in grads)>0
        assert not index.vectors.requires_grad
    finally:
        torch.set_num_threads(previous)


def test_missing_index_preflight_is_explicit(tmp_path):
    rows,_,_=run.resolve(seeds=[42],index_root=tmp_path/'missing')
    with pytest.raises(ValueError):
        run.validate_index(rows[0])
