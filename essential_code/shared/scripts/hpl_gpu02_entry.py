"""HPL entry with bounded native allocation policy and observational telemetry."""
import json
import os
from pathlib import Path
import runpy
import sys
import time

# glibc's adaptive mmap threshold can send 19 MiB image batches to the
# brk heap. Small epoch-long label arrays then strand freed batch-sized holes.
# Keep large temporaries independently releasable. This must precede allocator
# initialization, so direct invocations re-exec before importing torch/PIL.
if os.environ.get('MALLOC_MMAP_THRESHOLD_') != '131072':
    env = os.environ.copy()
    env['MALLOC_MMAP_THRESHOLD_'] = '131072'
    os.execve(sys.executable, [sys.executable, __file__, *sys.argv[1:]], env)

ROOT=Path(__file__).resolve().parents[1]
TRAINER=ROOT/'training/hpl_gpu02.py'
UPSTREAM=ROOT/'third_party/Heteroscedastic-Pseudo-Labels/utkface'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(UPSTREAM))
out=Path(sys.argv[sys.argv.index('--benchmark_output_dir')+1])
limit=int(os.environ.get('HPL_CANARY_STEPS','0'))
samples=[]
steps=0

def observer(epoch_step):
    global steps
    steps+=1
    if steps%10:return
    status=dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    rss=int(status['VmRSS'].split()[0])*1024
    row={'step':steps,'epoch_step':epoch_step,'unix':time.time(),'rss_bytes':rss}
    if steps % 100 == 0:
        rollup = dict(line.split(':', 1) for line in Path('/proc/self/smaps_rollup').read_text().splitlines()[1:])
        row.update(pss_bytes=int(rollup['Pss'].split()[0])*1024,
                   anonymous_bytes=int(rollup['Anonymous'].split()[0])*1024)
    with (out/'rss_iterations.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    if limit:
        samples.append(row)
        if steps>=limit:
            warm=[r for r in samples if r['step']>=50]
            growth=warm[-1]['rss_bytes']-warm[0]['rss_bytes']
            slope=growth/(warm[-1]['step']-warm[0]['step'])
            passed=max(r['rss_bytes'] for r in samples)<8*1024**3 and growth<512*1024**2 and slope<1024**2
            result={'status':'pass' if passed else 'failed','steps':steps,'warmup_steps':50,'rss_growth_bytes':growth,'rss_slope_bytes_per_step':slope,'peak_rss_bytes':max(r['rss_bytes'] for r in samples),'samples':samples,'formal_result':False}
            (out/'canary_result.json').write_text(json.dumps(result,indent=2)+'\n')
            raise SystemExit(0 if passed else 86)

runpy.run_path(str(TRAINER),run_name='__main__',init_globals={'iteration_observer':observer})
