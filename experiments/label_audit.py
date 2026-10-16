"""Audit folder targets against released final disease/severity sidecars.

No relabeling or exclusion is performed. A negative annotation for one disease
does not certify an ocularly healthy animal or an exhaustive negative exam.
"""
import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .common import ROOT,sha256,write_json,status

DISEASES={
    'Blepharitis':{'안검염'},'Cataract':{'백내장'},'Conjunctivitis':{'결막염'},
    'Entropion':{'안검내반증'},'Epiphora':{'유루증'},'Eyelid_tumor':{'안검종양'},
    'Non_ulcerative_keratitis':{'비궤양성각막질환','비궤양성각막염'},
    'Nuclear_sclerosis':{'핵경화'},'Pigmentary_keratitis':{'색소침착성각막염'},
    'Ulcerative_keratitis':{'궤양성각막질환','각막궤양'}}
POSITIVE={'유','상','하','초기','비성숙','성숙'}


def record(path):
    data=json.loads(path.read_text());label=data.get('label',{});folder=path.parent.name
    disease=str(label.get('label_disease_nm',''));final=str(label.get('label_disease_lv_3',''))
    if folder=='Normal':consistent=final=='무'
    else:consistent=disease in DISEASES[folder] and final in POSITIVE
    return dict(json_path=str(path.relative_to(ROOT/'data_raw')),**{'class':folder},
        disease=disease,final_level=final,reviewer1=str(label.get('label_disease_lv_1','')),
        reviewer2=str(label.get('label_disease_lv_2','')),deleted=str(label.get('label_deleted','')),
        expected_folder_rule=consistent,sidecar_sha256=sha256(path))


def run():
    status('label_semantics','IN PROGRESS')
    output=ROOT/'results/raw_label_annotations.csv'
    if output.exists():frame=pd.read_csv(output,keep_default_na=False)
    else:
        paths=sorted((ROOT/'data_raw').glob('*/*.json'))
        with ThreadPoolExecutor(max_workers=8) as pool:frame=pd.DataFrame(pool.map(record,paths))
        frame.to_csv(output,index=False)
    split=pd.read_csv(ROOT/'results/revision_split.csv',keep_default_na=False)
    split['json_path']=split.path.map(lambda p:str(__import__('pathlib').Path(p).with_suffix('.json')))
    retained=split[['json_path','split']].merge(frame,on='json_path',validate='one_to_one')
    if len(retained)!=len(split):raise ValueError('Released annotation coverage incomplete')
    result={}
    for scope,part in [('raw',frame),('retained',retained)]:
        distribution=part.groupby(['class','disease','final_level','deleted'],dropna=False).size().reset_index(name='images')
        distribution.to_csv(ROOT/'results/tables'/(scope+'_label_annotation_distribution.csv'),index=False)
        mismatches=part[~part.expected_folder_rule.astype(bool)]
        mismatches.to_csv(ROOT/'results/tables'/(scope+'_label_rule_disagreements.csv'),index=False)
        normal=part[part['class']=='Normal']
        result[scope]=dict(images=len(part),folder_rule_disagreements=len(mismatches),
            deleted_counts=part.deleted.value_counts().to_dict(),
            normal_final_level_counts=normal.final_level.value_counts().to_dict(),
            normal_annotated_disease_counts=normal.disease.value_counts().to_dict(),
            first_second_annotator_disagreement_count=int((part.reviewer1!=part.reviewer2).sum()))
    write_json(ROOT/'results/label_semantics_audit.json',dict(**result,
        original_sidecars_preserved=True,target_labels_changed=False,
        source_code_sha256=sha256(__file__),annotations_csv_sha256=sha256(output),
        split_sha256=sha256(ROOT/'results/revision_split.csv'),
        folder_rule=dict(normal='final disease severity is absence (무), not proof of all-disease absence',
                         positive_diseases={k:sorted(v) for k,v in DISEASES.items()},positive_levels=sorted(POSITIVE)),
        clinical_reference_standard='veterinarian-reviewed released final annotations; author confirms three veterinarians completed dataset review; original adjudication records not supplied',
        normal_semantics='source-specific negative class; exhaustive ocular health cannot be established'))
    status('label_semantics','COMPLETED',raw_images=len(frame),retained_images=len(retained),
           label_disagreements=result['retained']['folder_rule_disagreements'])
    print('Label audit',result,flush=True)


if __name__=='__main__':run()
