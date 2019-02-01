import os
import pickle
import time

import execnet

from firebase.upload import update_status
from src.jobs import ssh
from src.jobs.server_manager import ServerManager


class JobManager:
    def __init__(self, config: dict, func: callable, job_start_callback: callable, job_end_callback: callable):
        self.config = config
        self.func = func
        self.job_start = job_start_callback
        self.job_end = job_end_callback

        self.server_manager = ServerManager(config['servers'])

        self.gateways = {}          # Started sessions
        self.channels = {}          # Started programs
        self.occupancy = {}         # {job: server}

        self.job_count = 0          # total count of jobs seen
        self.jobs = {}              # All active jobs

        self.queue = []             # Queue of all jobs.
        self.priority_queue = []    # Queue that empties before self.queue

    def queue_job(self, parameters: tuple, priority: bool = False) -> int:
        id = self.__auto_id()
        self.jobs[id] = parameters

        if priority:
            self.priority_queue += [id]
        else:
            self.queue += [id]

        self.start_next_job()

    def start_next_job(self):
        priority_job = False
        if len(self.priority_queue) > 0:
            job_id = self.priority_queue.pop(0)
            priority_job = True
        elif len(self.queue) > 0:
            job_id = self.queue.pop(0)
        else:
            return

        # Add server job runs at to parameters:
        server, gateway = self.server_manager.create_gateway()
        if server and gateway:
            self.server_manager.servers[server['name']]['running jobs'] += 1
            self.jobs[job_id] += tuple([server])

            # Syncronizing models:
            self.__sync_files(job_id, upload=True)

            # Executing script and getting process id:
            channel = gateway.remote_exec(self.func)
            channel.setcallback(self.__finish_job(job_id))

            # Sending data to be processed and recieving fitness:
            args = tuple([pickle.dumps(arg) for arg in self.jobs[job_id]]) + tuple([job_id])
            channel.send(args)

            self.job_start(*self.jobs[job_id])
            self.gateways[job_id] = gateway
            self.channels[job_id] = channel
        else:
            if priority_job:
                self.priority_queue.insert(0, job_id)
            else:
                self.queue.insert(0, job_id)

    def __finish_job(self, job_id):
        # is a class method a valid callback?
        job_id = job_id
        this = self
        def finish(results):
            try:
                args = this.jobs[job_id]
                this.__sync_files(job_id, download=True)
                this.job_end(this, args, results)
                this.channels[job_id].close()
                this.gateways[job_id].exit()
                server = args[2]
                this.server_manager.servers[server['name']]['running jobs'] -= 1
                del this.jobs[job_id], this.channels[job_id], this.gateways[job_id]
            except:
                print(results)
            finally:
                this.start_next_job()
        return finish

    def await_all_jobs_finish(self):
        def print_progress(remaining, this):
            for i in range(remaining - (len(this.queue) + len(this.priority_queue))):
                print("=", end="", flush=True)

        total_jobs = len(self.queue) + len(self.priority_queue)
        remaining_jobs = total_jobs
        print("--> Finishing {} job(s): |".format(remaining_jobs), end="", flush=True)
        while len(self.queue) + len(self.priority_queue) > 0:
            print_progress(remaining_jobs, self)
            if remaining_jobs - (len(self.queue) + len(self.priority_queue)) > 0:
                done = total_jobs - remaining_jobs
                update_status("Completed {}/{} training sessions".format(done, total_jobs))
            remaining_jobs = (len(self.queue) + len(self.priority_queue))
            time.sleep(3)

        print_progress(total_jobs, self)
        print("|")

        if len(self.channels) > 0:
            mch = execnet.MultiChannel(list(self.channels.values()))
            mch.waitclose()
        if len(self.gateways) > 0:
            for gw in self.gateways.values():
                gw.exit()

    def __auto_id(self):
        id = self.job_count
        self.job_count += 1
        return id

    def __sync_files(self, job_id: int, download: bool = False, upload: bool = False):
        def download_models(individ, config, server_config):
            ssh.rsync(
                os.path.join(os.getcwd(), individ.relative_save_path(config)),
                os.path.join(server_config['cwd'], individ.relative_save_path(config)),
                server_config,
                to_source=True
            )

        def upload_models(individ, config, server_config):

            local_path = os.path.join(os.getcwd(), individ.relative_save_path(config))
            server_path = os.path.join(server_config['cwd'], individ.relative_save_path(config))
            ssh.rsync(local_path, server_path, server_config, to_source=False)
            if individ.predecessor:
                local_path = os.path.join(os.getcwd(), individ.predecessor.relative_save_path(config))
                server_path = os.path.join(server_config['cwd'],
                                           individ.predecessor.relative_save_path(config))
                ssh.rsync(local_path, server_path, server_config, to_source=False)

        individ, config, server_config = self.jobs[job_id]

        if server_config['type'] != 'local':
            if download:
                download_models(individ, config, server_config)
            if upload:
                upload_models(individ, config, server_config)
                if individ.predecessor:
                    upload_models(individ, config, server_config)
