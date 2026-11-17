from pathlib import PurePosixPath,Path
import io,json,tarfile,hashlib
import torch
from fusion.next.common import OUT,digest,source_identity,write_json,utc_now
from fusion.next.panel import Panel,CoverageSampler
from fusion.next.experiment import model_for
from fusion.next.trainer import optimizer_for
folder=OUT/'cloud/incremental_backups';receipt=json.loads((folder/'checkpoint_latest_receipt.json').read_text());path=folder/Path(receipt['archive']).name
if digest(path)!=receipt['sha256']:raise RuntimeError('Archive mismatch')
panel=Panel.load();verified=[]
with tarfile.open(path) as archive:
 members=archive.getmembers();names=[m.name for m in members]
 if len(names)!=len(set(names)) or any(not m.isfile() or PurePosixPath(m.name).is_absolute() or '..' in PurePosixPath(m.name).parts for m in members):raise RuntimeError('Unexpected archive member')
 manifest=json.loads(archive.extractfile('checkpoint_snapshot.json').read())
 if manifest['source_identity']!=source_identity() or set(names)!={*manifest['files'],'checkpoint_snapshot.json'}:raise RuntimeError('Manifest identity mismatch')
 for name,row in manifest['files'].items():
  data=archive.extractfile(name).read()
  if hashlib.sha256(data).hexdigest()!=row['sha256'] or len(data)!=row['bytes']:raise RuntimeError('Member mismatch')
 for row in manifest['portable_checkpoints']:
  saved=torch.load(io.BytesIO(archive.extractfile(row['path']).read()),map_location='cpu',weights_only=True)
  r=json.loads(archive.extractfile(row['path']+'.json').read());recipe=saved['recipe']
  if saved['step']!=r['step'] or recipe!=r['recipe']:raise RuntimeError('Checkpoint contents differ from receipt')
  if recipe['source_identity']!=source_identity():raise RuntimeError('Source mismatch')
  if not {'model','optimizer','samplers','rng','extra','recipe','step','schema'}.issubset(saved):raise RuntimeError('Recovery state incomplete')
  if saved['step'] and not saved['optimizer']['state']:raise RuntimeError('Trained checkpoint lacks optimizer state')
  model=model_for(panel,recipe['config'],'cpu');model.load_state_dict(saved['model'],strict=True)
  optimizer=optimizer_for(model,recipe['config'],'cpu');optimizer.load_state_dict(saved['optimizer'])
  for state in saved['samplers'].values():
   sampler=CoverageSampler(panel,recipe['fit_end'],state['seed'])
   sampler.load_state_dict(state);stocks,rows=sampler.sample(16)
   if len(rows)!=16 or any(panel.dates[rows]>recipe['fit_end']):raise RuntimeError('Restored sampler violates fit boundary')
  verified.append({'path':row['path'],'step':saved['step'],'tensor_state_entries':len(saved['model']),
   'actual_native_model_and_optimizer_loaded':True,'actual_coverage_sampler_restored_and_advanced':True})
write_json(folder/(receipt['sha256']+'_verification.json'),{'archive_sha256':receipt['sha256'],'verified_at_utc':utc_now(),'source_identity':source_identity(),'loaded_with_weights_only':True,'all_hashes_match':True,'verified_checkpoints':verified,'canonical_training_files_changed':False,'Q3_accessed':False})
print({'verified_checkpoint_count':len(verified),'archive_sha256':receipt['sha256']})
