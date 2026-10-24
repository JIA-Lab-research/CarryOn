"""Create one version-pinned Linux environment in a new directory."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def main():
    manifest=json.loads((ROOT/'requirements/environments.json').read_text())
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',required=True,choices=sorted(manifest['profiles']))
    p.add_argument('--python',required=True,help='Base Python executable for the specified profile')
    p.add_argument('--output',required=True,help='New environment directory; existing directories are refused')
    p.add_argument('--plan-only',action='store_true')
    args=p.parse_args();spec=manifest['profiles'][args.profile];versions=manifest['environments'][args.profile]
    base=shutil.which(args.python)
    if not base:raise ValueError('Python executable not found: '+args.python)
    env=dict(os.environ)
    for key in ('PYTHONPATH','PYTHONHOME','OPENROUTER_API_KEY','HF_TOKEN','HUGGING_FACE_HUB_TOKEN'):env.pop(key,None)
    env.update(PYTHONNOUSERSITE='1',PIP_INDEX_URL='https://pypi.org/simple')
    actual=json.loads(subprocess.check_output([base,'-c','import sys,json;print(json.dumps(list(sys.version_info[:3])))'],env=env,text=True))
    expected=[int(v) for v in versions['python'].split('.')[:2]]
    if actual[:2]!=expected:raise ValueError('This profile requires Python '+'.'.join(map(str,expected)))
    if sys.platform!='linux':raise ValueError('The simulation/model environments require Linux')
    output=Path(args.output).expanduser().absolute()
    if output.exists() or output.is_relative_to(ROOT):raise ValueError('Use a new directory outside the source tree')
    lock=ROOT/versions['requirements']
    plan=dict(profile=args.profile,python=actual,output=str(output),sources=[manifest['sources'][k] for k in spec['sources']],requirements=versions['requirements'])
    if args.plan_only:print(json.dumps(plan,indent=2));return
    output.mkdir(parents=True);(output/'PLAN.json').write_text(json.dumps(plan,indent=2))
    steps=[]
    def run(argv):
        steps.append(argv)
        with (output/('step_%02d.log'%len(steps))).open('xb') as log:
            result=subprocess.run(argv,cwd=output,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Environment setup failed; see '+str(output/('step_%02d.log'%len(steps))))
    try:
        # Preserve the base executable's native-library search paths; packages stay isolated in the new venv.
        run([base,'-m','venv','--symlinks',str(output/'venv')]);python=str(output/'venv/bin/python')
        run([python,'-m','pip','install','--no-deps','-r',str(lock)])
        sources=output/'sources';sources.mkdir()
        for key in spec['sources']:
            source=manifest['sources'][key];target=sources/key
            run(['git','init',str(target)])
            run(['git','-C',str(target),'remote','add','origin',source['url']])
            run(['git','-C',str(target),'fetch','--depth','1','origin',source['revision']])
            run(['git','-C',str(target),'checkout','--detach','FETCH_HEAD'])
            commit=subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],env=env,text=True).strip()
            if commit!=source['revision']:raise ValueError('Source revision mismatch: '+key)
        for item in spec['install']:
            source=sources/item['source']/item.get('subdirectory','')
            run([python,'-m','pip','install','--no-deps','--no-build-isolation',str(source)])
        if args.profile in manifest['binary_wheels']:
            wheel=manifest['binary_wheels'][args.profile]
            target=output/urllib.parse.unquote(wheel['url'].rsplit('/',1)[-1])
            with urllib.request.urlopen(wheel['url'],timeout=120) as source,target.open('xb') as dest:shutil.copyfileobj(source,dest)
            run([python,'-m','pip','install','--no-deps',str(target)])
        if spec.get('transformers_patch'):
            records=spec['transformers_patch']
            target=Path(subprocess.check_output([python,'-c','import transformers;print(transformers.__path__[0])'],env=env,text=True).strip())
            source=sources/'openpi/src/openpi/models_pytorch/transformers_replace'
            for row in records:
                src=source/row;dst=target/row
                dst.parent.mkdir(parents=True,exist_ok=True);temp=dst.with_suffix(dst.suffix+'.carryon-new');temp.write_bytes(src.read_bytes());os.replace(temp,dst)
        run([python,'-m','pip','check'])
        if args.profile in ('simulation','safety-client','starvla','oft'):
            env.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
            run([python,'-B',str(ROOT/'src/carryon/experiments/safety/kinematics.py')])
        result=dict(status='completed',profile=args.profile,python=python,commands=steps)
    except BaseException as exc:
        (output/'INCOMPLETE.json').write_text(json.dumps(dict(status='incomplete',error=type(exc).__name__,message=str(exc),commands=steps),indent=2));raise
    (output/'RESULT.json').write_text(json.dumps(result,indent=2))
    print('Prepared',args.profile,'at',python)

if __name__=='__main__':main()
