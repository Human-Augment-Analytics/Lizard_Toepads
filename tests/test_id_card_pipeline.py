"""Synthetic verification; real images, a GPU, and Tesseract are not needed."""
from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('card_pipeline', ROOT/'scripts/id_cards/pipeline.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class CardPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images, self.annotations = self.root/'images', self.root/'annotations'
        self.images.mkdir(); self.annotations.mkdir()
        self.manifest = self.root/'splits.csv'
        self.template = self.root/'fields.json'
        self.template.write_text(json.dumps({'fields': {'species': [.1,.2,.8,.4]}}))
        self.cfg = {'images': str(self.images), 'annotations': str(self.annotations),
                    'manifest': str(self.manifest), 'output': str(self.root/'output'),
                    'template': str(self.template), 'card_size': [80,120], 'scan_max_side': 128,
                    'train': {'model': 'yolo11m-obb.pt','device':0,'epochs':200,'batch':4,'imgsz':1280},
                    'ocr': {'executable':'tesseract','language':'eng','psm':7,'truth':None,
                            'detection_conf':.25}}
        self.original = np.array([[30,40],[130,40],[130,240],[30,240]],float)
        self.record = {'image_id':'0000','target_resolution':1024,
              'transform': [[1.5,0,8],[0,1.5,6]],
              'transform_inverse': [[1/1.5,0,-8/1.5],[0,1/1.5,-6/1.5]],
              'obb_corners': (self.original*1.5+[8,6]).tolist()}
        self.rows = []
        for i in range(92):
            image_id = f'{i:04d}'
            Image.new('RGB',(200,300),'white').save(self.images/f'{image_id}.jpg')
            rec = dict(self.record,image_id=image_id)
            (self.annotations/f'{image_id}.json').write_text(json.dumps(rec))
            split = 'train' if i<60 else 'val' if i<70 else 'test' if i<90 else 'field'
            self.rows.append({'image_id':image_id,'group_id':f'animal{i}','split':split,
                              'rotation_quarters':'0'})
        self.write_manifest()

    def write_manifest(self):
        with self.manifest.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['image_id','group_id','split','rotation_quarters'])
            w.writeheader();w.writerows(self.rows)

    def test_inverse_transform_and_invalid_geometry(self):
        np.testing.assert_allclose(p.corners(self.record,(200,300)),self.original)
        bad=deepcopy(self.record);bad['transform_inverse'][0][0]=3
        with self.assertRaisesRegex(ValueError,'transforms disagree'):p.corners(bad,(200,300))
        bad=deepcopy(self.record);bad['obb_corners'][0]=[9999,9999]
        with self.assertRaisesRegex(ValueError,'outside'):p.corners(bad,(200,300))
        bad=deepcopy(self.record);bad['obb_corners']=[[20,20]]*4
        with self.assertRaisesRegex(ValueError,'degenerate'):p.corners(bad,(200,300))

    def test_manifest_limits_and_group_leakage(self):
        images,records=p.inventory(self.cfg)
        self.assertEqual(len(p.manifest(self.cfg,images,records)),92)
        self.rows[60]['group_id']=self.rows[0]['group_id'];self.write_manifest()
        with self.assertRaisesRegex(ValueError,'leaks'):p.manifest(self.cfg,images,records)
        self.rows[60]['group_id']='animal60'
        for row in self.rows[:20]:row['split']='val'
        self.write_manifest()
        with self.assertRaisesRegex(ValueError,'50–100'):p.manifest(self.cfg,images,records)

    def test_missing_images_fail_without_substituting_tps_scans(self):
        (self.images/'0000.jpg').unlink()
        with self.assertRaisesRegex(ValueError,'lack matching images'):p.inventory(self.cfg)

    def test_prepare_one_class_and_baseline_crop_without_yolo(self):
        p.prepare(self.cfg)
        out=p.resolve(self.cfg['output'])
        self.assertTrue((out/'READY').exists())
        data=yaml.safe_load((out/'dataset.yaml').read_text())
        self.assertEqual(data['names'],{0:'id_card'})
        label=(out/'labels/train/0000.txt').read_text().split()
        self.assertEqual(label[0],'0');self.assertEqual(len(label),9)
        np.testing.assert_allclose(np.array(label[1:],float).reshape(4,2),self.original/[200,300])
        with Image.open(out/'cards/train/0000.png') as card:self.assertEqual(card.size,(80,120))
        p.baseline(types.SimpleNamespace(split='val',ocr=False),self.cfg)
        self.assertEqual(len(list((out/'baseline/val').glob('*_species.png'))),10)
        with Image.open(out/'baseline/val/0060_species.png') as field:self.assertEqual(field.size,(56,24))
        with self.assertRaises(FileExistsError):p.prepare(self.cfg)

    def test_empty_and_invalid_template_rejected(self):
        self.template.write_text('{"fields":{}}')
        with self.assertRaisesRegex(ValueError,'empty'):p.fields(self.cfg)
        self.template.write_text('{"fields":{"species":[0.5,0.2,0.1,0.4]}}')
        with self.assertRaisesRegex(ValueError,'Empty field'):p.fields(self.cfg)

    def test_ocr_metrics_include_detection_miss(self):
        truth=self.root/'truth.csv';truth.write_text('image_id,field,value\n0001,species,Sag\n0002,species,Sag\n')
        predictions=[{'image_id':'0001','field':'species','text':' SAG ','error':''},
                     {'image_id':'0002','field':'species','text':'','error':'card detection missed'}]
        p.score(predictions,truth,self.root,scope='full pipeline')
        metrics=json.loads((self.root/'metrics.json').read_text())
        self.assertEqual(metrics['normalized_exact_accuracy'],.5)
        self.assertEqual(metrics['character_error_rate'],.5)
        self.assertEqual(p.distance('abc','axc'),1)

    def test_gpu_smoke_and_yolo_test_are_separate(self):
        out=p.resolve(self.cfg['output']);out.mkdir()
        (out/'READY').write_text('done');(out/'config.yaml').write_text(yaml.safe_dump(self.cfg))
        calls=[]
        class FakeModel:
            task='obb'
            names={0:'id_card'}
            def __init__(self,*a,**k):pass
            def train(self,**kw):
                calls.append(('train',kw));return types.SimpleNamespace(save_dir=str(out))
            def val(self,**kw):
                calls.append(('val',kw));return types.SimpleNamespace(save_dir=str(out),results_dict={'mAP':.5})
        torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:True))
        with patch.dict('sys.modules',{'torch':torch,'ultralytics':types.SimpleNamespace(YOLO=FakeModel)}):
            p.model_action(types.SimpleNamespace(command='smoke',resume=None),self.cfg)
            self.assertEqual(calls[-1][1]['epochs'],2)
            self.assertEqual(calls[-1][1]['data'],str(out/'dataset.yaml'))
            checkpoint=self.root/'best.pt';checkpoint.touch()
            p.model_action(types.SimpleNamespace(command='test',weights=str(checkpoint)),self.cfg)
            self.assertEqual(calls[-1][1]['split'],'test')
            torch.cuda.is_available=lambda:False
            with self.assertRaisesRegex(RuntimeError,'GPU'):p.model_action(types.SimpleNamespace(command='train'),self.cfg)

    def test_field_detection_order_and_rotation(self):
        pts=np.array([[80,110],[10,10],[10,110],[80,10]],np.float32)
        np.testing.assert_allclose(p.detection_order(pts),[[10,10],[80,10],[80,110],[10,110]])
        image=Image.new('RGB',(200,300),'white')
        card=p.rectify(image,self.original,[80,120],turns=2)
        self.assertEqual(card.size,(80,120))

    def test_field_pipeline_scores_detector_misses(self):
        p.prepare(self.cfg)
        truth=self.root/'truth.csv'
        truth.write_text('image_id,field,value\n0090,species,Sag\n0091,species,Sag\n')
        self.cfg['ocr']['truth']=str(truth)
        checkpoint=self.root/'best.pt';checkpoint.touch()
        class FakeModel:
            task='obb'
            names={0:'id_card'}
            def __init__(self,*a,**k):pass
            def predict(self,**kw):return [types.SimpleNamespace(obb=None)]
        torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:True))
        with patch.dict('sys.modules',{'torch':torch,'ultralytics':types.SimpleNamespace(YOLO=FakeModel)}), \
             patch.object(p.shutil,'which',return_value='/synthetic/tesseract'):
            p.field_evaluate(types.SimpleNamespace(weights=str(checkpoint)),self.cfg)
        out=p.resolve(self.cfg['output'])/'field_evaluation'
        self.assertEqual(json.loads((out/'detection_metrics.json').read_text())['missed_cards'],2)
        self.assertEqual(json.loads((out/'metrics.json').read_text())['normalized_exact_accuracy'],0)

    def test_retirement_preserves_existing_data_and_archives_config(self):
        config=self.root/'configs';config.mkdir()
        old=config/'oneal_obb.yaml';old.write_text('local edited config')
        data=self.root/'data/obb/oneal_v1';data.mkdir(parents=True)
        (data/'checkpoint.pt').write_text('existing checkpoint')
        with patch.object(p,'ROOT',self.root):p.retire()
        self.assertFalse(old.exists())
        self.assertEqual((self.root/'data/obb/retired-configs/oneal_obb.yaml').read_text(),'local edited config')
        self.assertEqual((data/'checkpoint.pt').read_text(),'existing checkpoint')


    def test_manual_baseline_needs_no_prepared_yolo_dataset(self):
        self.manifest.unlink()
        p.baseline(types.SimpleNamespace(split='val',ocr=False,image_id=['0000'],rotation_quarters=0),self.cfg)
        out=p.resolve(self.cfg['output'])
        self.assertTrue((out.parent/(out.name+'_manual_baseline')/'0000_species.png').exists())
        self.assertFalse((out/'READY').exists())



if __name__=='__main__':unittest.main()
