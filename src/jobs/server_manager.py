import execnet


class ServerManager:

    def __init__(self, servers):
        self.all_servers = servers
        self.servers = {}
        for server in servers:
            server['running jobs'] = 0
            self.servers['name'] = server

    def create_gateway(self):
        server = self.__get_free_server()
        if server:
            gateway = self.__connect(server)
            return server, gateway
        else:
            return None, None

    def __connect(self, server):
        def get_local_gateway(server):
            return execnet.makegateway("ssh={ssh}//python={py}//chdir={dir}".format(
                ssh="{}@{}".format(server['username'], server['address']),
                py="python",
                dir=server['cwd'],
            ))

        if server['type'] == 'ssh':
            return get_local_gateway(server)
        else:
            raise AttributeError('Unknown server "type": ' + server['type'])

    def __get_free_server(self):
        free_servers = sorted([
            server for server in self.all_servers
            if server['running jobs'] < server['concurrency']
        ], key=lambda s: s['running jobs'])

        if free_servers:
            return free_servers[0]