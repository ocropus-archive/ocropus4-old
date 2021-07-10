from invoke import task
import yaml
import io
import os
import re
import inspect
import time
import json

authfile = "gceauth.json"
authinfo = json.load(open(authfile))
project = authinfo["project_id"]
numnodes = 64
zone = "us-west1-a"
machinetype = "n1-standard-8"
image = "o4extract"


@task
def buildlocal(c):
    c.run("cd ../.. && git archive HEAD --prefix=ocropus4/ -o training/runextract/docker/ocropus4.tar.gz")
    c.run(f"cd docker && docker build -t o4extract .")


@task
def build(c):
    buildlocal(c)
    c.run(f"docker tag {image} tmbdev/{image}")
    c.run(f"docker push tmbdev/{image}")
    c.run(f"docker tag {image} gcr.io/{project}/{image}")
    c.run(f"docker push gcr.io/{project}/{image}")


@task
def gcestart(c):
    assert os.path.exists(authfile)
    c.run("gcloud config set project {project}")
    c.run("gcloud config set compute/zone {zone}")
    c.run(
        f"gcloud container clusters create {image} "
        + "--machine-type=$machinetype "
        + "--scopes storage-rw,compute-rw "
        + "--num-nodes=$numnodes"
    )
    c.run(f"gcloud container clusters get-credentials {image}")


@task
def gcestop(c):
    c.run(f"yes | gcloud container clusters delete {image}")


job_template = f"""
apiVersion: v1
kind: Pod
metadata:
  name: "__NAME__"
  labels:
    app: ubuntu-app
spec:
  containers:
  - name: tmbdev-extract
    image: tmbdev/{image}
    command: ["/bin/bash", "-c", "date"]
    resources:
      requests:
        cpu: 2
        memory: 8G
      limits:
        cpu: 2
        memory: 8G
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
        job = submit_script(name, word_script, mem="16G")
        with os.popen("kubectl apply -f -", "w") as stream:
            stream.write(job)
        print(name, shard)
        time.sleep(1)
