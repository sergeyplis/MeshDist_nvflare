import subprocess
from nvflare.fuel.flare_api.flare_api import new_secure_session, Session
from nvflare.apis.job_def import RunStatus, JobMetaKey


def start_server():
    subprocess.run(["/bin/bash", "/runKit/server/startup/start.sh"],
                   cwd="/runKit/server/startup")


start_server()

session = new_secure_session(
    "admin@admin.com",
    "/runKit/admin/"
)

jobId = session.submit_job(
    "/workspace/jobs/job/"
)


def my_callback(session: Session, job_id: str, job_meta, *cb_args, **cb_kwargs) -> bool:
    job_status = job_meta[JobMetaKey.STATUS]
    print(f"job status: {job_status}")

    if 'FINISHED' in job_status:
        print(f"job {job_id} finished, shutting down system")
        session.shutdown("all")
        return False
    else:
        return True


session.monitor_job(
    jobId, timeout=3600, poll_interval=10, cb=my_callback)
