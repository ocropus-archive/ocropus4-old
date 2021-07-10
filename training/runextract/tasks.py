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
assert isinstance(project, str)
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
    c.run(f"gcloud config set project {project}")
    c.run(f"gcloud config set compute/zone {zone}")
    c.run(
        f"gcloud container clusters create {image} "
        + f"--num-nodes={numnodes} "
        + f"--machine-type={machinetype} "
        + "--scopes storage-rw,compute-rw "
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


@task
def status(c):
    c.run("kubectl get pods | grep -v STATUS | awk '{print $3}' | sort | uniq -c")


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


def runscript(c, src, dst, template):
    c.run("kubectl delete pods --all")
    sources = os.popen(f"gsutil ls {src}").readlines()
    sources = set(re.sub(".*/", "", s).strip() for s in sources)
    existing = os.popen(f"gsutil ls {dst}").readlines()
    existing = set(re.sub(".*/", "", s).strip() for s in existing)
    missing = sources.difference(existing)
    missing = sorted(list(missing))
    if len(missing) == 0:
        print("nothing to be done")
        return
    print(f"# submitting {len(missing)} jobs")
    print(f"# {missing[::max(len(missing)//10, 1)]}")
    print(f"# starting in 5 seconds")
    time.sleep(5)
    for index, shard in enumerate(missing):
        script = inspect.cleandoc(template.format(src=src, dst=dst, shard=shard, index=index))
        name = f"job-{index}"
        job = submit_script(name, script, mem="16G")
        with os.popen("kubectl apply -f -", "w") as stream:
            stream.write(job)
        print(name, shard)
        if index < 4:
            print()
            print(job)
            print()
        time.sleep(1)


@task
def words(c):
    runscript(
        c,
        "gs://nvdata-ocropus-tess",
        "gs://nvdata-ocropus-words",
        """
        gsutil cat "{src}/{shard}" |
        ocropus4 extract-rec hocr2rec --extensions "nrm.jpg;jpg;png;page.jpg;page.png hocr;hocr.html" - --output - |
        gsutil cp - "{dst}/{shard}"
    """,
    )


@task
def lines(c):
    runscript(
        c,
        "gs://nvdata-ocropus-tess",
        "gs://nvdata-ocropus-lines",
        """
        gsutil cat "{src}/{shard}" |
        ocropus4 extract-rec hocr2rec --bounds 40,40,3000,400 --element ocr_line --extensions "nrm.jpg;jpg;png;page.jpg;page.png hocr;hocr.html" - --output - |
        gsutil cp - "{dst}/{shard}"
    """,
    )


@task
def wseg(c):
    runscript(
        c,
        "gs://nvdata-ocropus-tess",
        "gs://nvdata-ocropus-wseg",
        """
        gsutil cat "{src}/{shard}" |
        ocropus4 extract-seg hocr2seg --check word - --output - |
        gsutil cp - "{dst}/{shard}"
    """,
    )


@task
def lseg(c):
    runscript(
        c,
        "gs://nvdata-ocropus-tess",
        "gs://nvdata-ocropus-lseg",
        """
        gsutil cat "{src}/{shard}" |
        ocropus4 extract-seg hocr2seg --check line --element ocr_line - --output - |
        gsutil cp - "{dst}/{shard}"
    """,
    )
