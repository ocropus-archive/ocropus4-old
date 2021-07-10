from invoke import task
import yaml
import io
import os
import re
import inspect
import time

job_template = """
apiVersion: v1
kind: Pod
metadata:
  name: "__NAME__"
  labels:
    app: ubuntu-app
spec:
  containers:
  - name: tmbdev-extract
    image: tmbdev/o4extract
    command: ["/bin/bash", "-c", "date"]
    resources:
      requests:
        cpu: 2
        memory: 4G
      limits:
        cpu: 2
        memory: 4G
  restartPolicy: Never
"""


script = """
echo hello
echo world
"""


def submit_script(name, script, cpu="2", mem="4G"):
    job = yaml.load(io.StringIO(job_template))
    job["metadata"]["name"] = name
    container = job["spec"]["containers"][0]
    container["command"][2] = script
    resources = container["resources"]
    resources["requests"]["cpu"] = cpu
    resources["requests"]["memory"] = mem
    resources["limits"]["cpu"] = cpu
    resources["limits"]["memory"] = mem
    job = yaml.dump(job)
    return job


@task
def getwords(c):
    c.run("kubectl delete pods --all")
    sources = os.popen("gsutil ls gs://nvdata-ocropus-tess").readlines()
    sources = set(re.sub(".*/", "", s).strip() for s in sources)
    existing = os.popen("gsutil ls gs://nvdata-ocropus-words").readlines()
    existing = set(re.sub(".*/", "", s).strip() for s in existing)
    missing = sources.difference(existing)
    print(f"# submitting {len(missing)} jobs")
    for index, shard in enumerate(sorted(missing)):
        word_script = inspect.cleandoc(
            f"""
            gsutil cat "gs://nvdata-ocropus-tess/{shard}" |
            ocropus4 extract-rec hocr2rec --extensions "nrm.jpg;jpg;png;page.jpg;page.png hocr;hocr.html" - --output - |
            gsutil cp - "gs://nvdata-ocropus-words/{shard}"
        """
        )
        name = f"words-{index}"
        job = submit_script(name, word_script)
        with os.popen("kubectl apply -f -", "w") as stream:
            stream.write(job)
        print(job)
        time.sleep(1)
