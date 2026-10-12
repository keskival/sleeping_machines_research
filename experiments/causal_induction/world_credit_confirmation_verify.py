"""Verify both immutable original and fresh confirmation packet schemas."""
import json
from world_credit_pipeline import ROOT,RESULTS,PARENT,sha

def verify_result(tag):
    result=json.loads((RESULTS/(tag+'.json')).read_text());assert result['status']=='completed'
    for name,digest in result['source_sha256'].items():assert sha(ROOT/name)==digest,name
    parents=result.get('parent_sha256',{})
    if isinstance(parents,str):assert sha(PARENT)==parents
    else:
        for name,digest in parents.items():assert sha(ROOT/name)==digest,name
    for row in result.get('data_records',result.get('records',[])):
        assert sha(ROOT/row['path'])==row['sha256'],row['path']
    return result
