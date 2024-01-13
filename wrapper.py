from CybORG.Agents.Wrappers.BaseWrapper import BaseWrapper
from CybORG.Agents.Wrappers.TrueTableWrapper import TrueTableWrapper
from CybORG.Shared.Results import Results
from CybORG.Shared.Actions import (
    Monitor,
    Analyse,
    Remove,
    Restore,
    PrivilegeEscalate,
    ExploitRemoteService,
    DiscoverRemoteSystems,
    Impact,
    DiscoverNetworkServices,
    Sleep,
)

from itertools import product
from copy import deepcopy
from prettytable import PrettyTable
import numpy as np

class CompetitiveWrapper(BaseWrapper):
    def __init__(self, turns, env=None, agent=None, output_mode="vector"):
        super().__init__(env, agent)
        self.env = TrueTableWrapper(env=env, agent=agent)
        self.agent = agent

        self.red_info = {}
        self.known_subnets = set()
        self.step_counter = -1
        self.id_tracker = -1
        self.output_mode = output_mode
        self.success = None

        self.subnets = "Op", "User"
        self.hostnames = (
            "Op_Host0",
            "Op_Server0",
            "User1",
            "User2",
            "User3",
        )

        blue_lone_actions = [["Monitor"]]  # actions with no parameters
        blue_host_actions = (
            "Analyse",
            "Remove",
            "Restore",
        )  # actions with a hostname parameter
        red_lone_actions = [["Sleep"], ["Impact"]]  # actions with no parameters
        red_network_actions = [
            "DiscoverSystems"
        ]  # actions with a subnet as the parameter
        red_host_actions = (
            "DiscoverServices",
            "ExploitServices",
            "PrivilegeEscalate",
        )
        self.blue_action_list = blue_lone_actions + list(
            product(blue_host_actions, self.hostnames)
        )
        self.red_action_list = (
            red_lone_actions
            + list(product(red_network_actions, self.subnets))
            + list(product(red_host_actions, self.hostnames))
        )

        self.subnet_map = {}  # subnets are ordered [Op, User]
        self.ip_map = (
            {}
        )  # ip addresses are ordered ['Op_Host0', 'Op_Server0', 'User0', 'User1', 'User2', 'User3']

        self.host_scan_status = [0]*len(self.hostnames)
        self.subnet_scan_status = [0]*len(self.subnets)*2
        # see _create_red_vector for explanation of flags
        self.impact_status = [0]

        self.turns_per_game = turns
        self.turn = 0
    
    # convert the discrete action choice into its corresponding CybORG action
    def resolve_blue_action(self, action):
        # assume a "single session" in the CybORG action space
        cyborg_space = self.get_action_space(agent="Blue")
        session = list(cyborg_space["session"].keys())[0]

        cyborg_action = self.blue_action_list[action]
        if cyborg_action[0] == "Analyse":
            return Analyse(hostname=cyborg_action[1], agent="Blue", session=session)
        elif cyborg_action[0] == "Remove":
            return Remove(hostname=cyborg_action[1], agent="Blue", session=session)
        elif cyborg_action[0] == "Restore":
            return Restore(hostname=cyborg_action[1], agent="Blue", session=session)
        else:
            return Monitor(agent="Blue", session=session)
    
    def resolve_red_action(self, action):
        # assume a single session in the cyborg action space
        cyborg_space = self.get_action_space(agent="Red")
        session = list(cyborg_space["session"].keys())[0]

        cyborg_action = self.red_action_list[action]
        if cyborg_action[0] == "Impact":
            return Impact(agent="Red", hostname="Op_Server0", session=session)
        elif cyborg_action[0] == "DiscoverSystems":
            return DiscoverRemoteSystems(
                subnet=self.subnet_map[cyborg_action[1]],
                agent="Red",
                session=session,
            )
        elif cyborg_action[0] == "DiscoverServices":
            return DiscoverNetworkServices(
                ip_address=self.ip_map[cyborg_action[1]],
                agent="Red",
                session=session,
            )
        elif cyborg_action[0] == "ExploitServices":
            return ExploitRemoteService(
                ip_address=self.ip_map[cyborg_action[1]],
                agent="Red",
                session=session,
            )
        elif cyborg_action[0] == "PrivilegeEscalate":
            return PrivilegeEscalate(
                hostname=cyborg_action[1], agent="Red", session=session
            )
        else:
            return Sleep()

    def map_network(self, env):

        i = 0  # count through the networks to assign the correct IP
        for subnet in env.get_action_space(agent="Red")["subnet"]:
            self.subnet_map[self.subnets[i]] = subnet
            i += 1

        i = 0  # counter through the IP addresses to assign the correct hostname
        for address in env.get_action_space(agent="Red")["ip_address"]:
            # skip mapping the Defender client and User0
            # Defender is at index 0, User0 is at index 3
            if i < 3:
                if i != 0:
                    self.ip_map[self.hostnames[i - 1]] = address
            if i > 3:
                self.ip_map[self.hostnames[i - 2]] = address
            i += 1

    # returns the blue and red observation vectors
    def reset(self):
        self.blue_info = {}
        self.red_info = {}
        self.known_subnets = set()
        self.step_counter = -1
        self.id_tracker = -1
        self.success = None

        self.turn = 0

        result = self.env.reset()
        self.map_network(self.env)

        self.host_scan_status = [0]*len(self.hostnames)
        self.subnet_scan_status = [0]*len(self.subnets)*2
        self.impact_status = [0]

        blue_obs = result.blue_observation # the environment now returns both observations, so blue_observation needs to be specified here
        self._initial_blue_obs(blue_obs)
        blue_vector = self.blue_observation_change(blue_obs, baseline=True)

        red_obs = result.red_observation
        red_vector = self.red_observation_change(red_obs, self.get_last_action(agent="Red"))

        return (blue_vector, red_vector)
    
    def step(self, red_action, blue_action) -> Results:

        red_step = self.resolve_red_action(red_action)
        blue_step = self.resolve_blue_action(blue_action)

        result = self.env.step(red_step, blue_step)
        self.turn += 1

        blue_obs = result.blue_observation
        blue_vector = self.blue_observation_change(blue_obs)
        result.blue_observation = blue_vector

        red_obs = result.red_observation
        red_vector = self.red_observation_change(red_obs, self.get_last_action(agent="Red"))
        result.red_observation = red_vector   

        result.action_space = self.action_space_change(result.action_space)
        # note this is the blue reward leaving the wrapper, red trainer must flip signal

        return result


    def _initial_blue_obs(self, obs):
        obs = obs.copy()
        self.baseline = obs
        del self.baseline["success"]
        for hostid in obs:
            if hostid == "success":
                continue
            host = obs[hostid]
            interface = host["Interface"][0]
            subnet = interface["Subnet"]
            ip = str(interface["IP Address"])
            hostname = host["System info"]["Hostname"]
            self.blue_info[hostname] = [str(subnet), str(ip), hostname, "None", "No"]
        return self.blue_info
    
    def blue_observation_change(self, observation, baseline=False):
        obs = observation if type(observation) == dict else observation.data
        obs = deepcopy(observation)
        success = obs["success"]

        self._process_blue_action(success)
        anomaly_obs = self._detect_anomalies(obs) if not baseline else obs
        del obs["success"]
        # TODO check what info is for baseline
        info = self._process_anomalies(anomaly_obs)
        if baseline:
            for host in info:
                info[host][-2] = "None"
                info[host][-1] = "No"
                self.blue_info[host][-1] = "No"

        self.info = info

        if self.output_mode == "table":
            return self._create_blue_table(success)
        elif self.output_mode == "anomaly":
            anomaly_obs["success"] = success
            return anomaly_obs
        elif self.output_mode == "raw":
            return observation
        elif self.output_mode == "vector":
            return self._create_blue_vector(success)
        else:
            raise NotImplementedError("Invalid output_mode for BlueTableWrapper")

    def _process_blue_action(self, success):
        action = self.get_last_action(agent="Blue")
        if action is not None:
            name = action.__class__.__name__
            hostname = (
                action.get_params()["hostname"]
                if name in ("Restore", "Remove")
                else None
            )

            if name == "Restore":
                self.blue_info[hostname][-1] = "No"
                # Update Red Access, if Red is aware of this machine
                ip = str(self.ip_map[hostname])
                if ip in self.red_info:
                    self.red_info[ip][4] = "None"

            # Old observation update for removing, superceded by moving any User machines to Unknow below
            elif name == "Remove":
                compromised = self.blue_info[hostname][-1]
                if compromised != 'No':
                    self.blue_info[hostname][-1] = 'Unknown'
                # Update Red Access, if Red is aware of this machine
                ip = str(self.ip_map[hostname])
                if ip in self.red_info:
                    if self.red_info[ip][4] == "User":
                        self.red_info[ip][4] = "None"
    
    def _detect_anomalies(self, obs):
        if self.baseline is None:
            raise TypeError(
                "BlueTableWrapper was unable to establish baseline. This usually means the environment was not reset before calling the step method."
            )

        anomaly_dict = {}

        for hostid, host in obs.items():
            if hostid == "success":
                continue

            host_baseline = self.baseline[hostid]
            if host == host_baseline:
                continue

            host_anomalies = {}
            if "Files" in host:
                baseline_files = host_baseline.get("Files", [])
                anomalous_files = []
                for f in host["Files"]:
                    if f not in baseline_files:
                        anomalous_files.append(f)
                if anomalous_files:
                    host_anomalies["Files"] = anomalous_files

            if "Processes" in host:
                baseline_processes = host_baseline.get("Processes", [])
                anomalous_processes = []
                for p in host["Processes"]:
                    if p not in baseline_processes:
                        anomalous_processes.append(p)
                if anomalous_processes:
                    host_anomalies["Processes"] = anomalous_processes

            if host_anomalies:
                anomaly_dict[hostid] = host_anomalies

        return anomaly_dict
    
    def _process_anomalies(self, anomaly_dict):
        info = deepcopy(self.blue_info)
        for hostid, host_anomalies in anomaly_dict.items():
            assert len(host_anomalies) > 0
            if "Processes" in host_anomalies:
                connection_type = self._interpret_connections(
                    host_anomalies["Processes"]
                )
                info[hostid][-2] = connection_type
                if connection_type == "Exploit":
                    info[hostid][-1] = "User"
                    self.blue_info[hostid][-1] = "User"

            if "Files" in host_anomalies:
                malware = [f["Density"] >= 0.9 for f in host_anomalies["Files"]]
                if any(malware):
                    info[hostid][-1] = "Privileged"
                    self.blue_info[hostid][-1] = "Privileged"

        return info

    def _interpret_connections(self, activity: list):
        num_connections = len(activity)

        ports = set(
            [
                item["Connections"][0]["local_port"]
                for item in activity
                if "Connections" in item
            ]
        )
        port_focus = len(ports)

        remote_ports = set(
            [
                item["Connections"][0].get("remote_port")
                for item in activity
                if "Connections" in item
            ]
        )
        if None in remote_ports:
            remote_ports.remove(None)

        if num_connections >= 3 and port_focus >= 3:
            anomaly = "Scan"
        elif 4444 in remote_ports:
            anomaly = "Exploit"
        elif num_connections >= 3 and port_focus == 1:
            anomaly = "Exploit"
        elif "Service Name" in activity[0]:
            anomaly = "None"
        else:
            anomaly = "Scan"

        return anomaly
    
    def _create_blue_vector(self, success):
        table = self._create_blue_table(success)._rows

        proto_vector = []

        # five flags for each host:
        # 1 - is this host uncompromised
        # 2 - was this host exploited last turn
        # 3 - is the status of this host unknown (has been exploited, and not restored or analyzed)
        # 4 - does Red have at least User privleges (exploited, and never removed or restored)
        # 5 - does Red have Root privleges (host was analyzed)

        # for each of the 5 machines, check these flags and add them to the observation vector
        for row in table:
            if ((row[2] == "Defender") or (row[2] == "User0")):
                continue

            # Flag 1
            compromised = row[4]
            if compromised == "No":
                value = [1]
            else:
                value = [0]
            proto_vector.extend(value)

            # Flag 2
            activity = row[3]
            if activity == "Exploit":
                value = [1]
            else:
                value = [0]
            proto_vector.extend(value)

            # Flag 3
            if ((compromised != "No") and (compromised != "Privileged")):
                value = [1]
            else:
                value = [0]
            proto_vector.extend(value)

            # Flag 4
            if compromised == "User":
                value = [1]
            else:
                value = [0]
            proto_vector.extend(value)

            # Flag 5
            if compromised == "Privileged":
                value = [1]
            else:
                value = [0]
            proto_vector.extend(value)
        
        turn_vector = [0]*(self.turns_per_game+1)
        turn_vector[self.turn] = 1
        proto_vector.extend(turn_vector)

        return np.array(proto_vector)
    
    def _create_blue_table(self, success):
        table = PrettyTable(
            ["Subnet", "IP Address", "Hostname", "Activity", "Compromised"]
        )
        for hostid in self.info:
            table.add_row(self.info[hostid])

        table.sortby = "Hostname"
        table.success = success
        return table
    
    def get_blue_table(self, output_mode="blue_table"):
        if output_mode == "blue_table":
            return self._create_blue_table(success=None)
        elif output_mode == "true_table":
            return self.env.get_table()

    def red_observation_change(self, observation, action):
        self.success = observation["success"]

        self.step_counter += 1
        if self.step_counter <= 0:
            self._initial_red_obs(observation)
        elif self.success:
            self._update_red_info(observation, action)

        if self.output_mode == "table":
            obs = self._create_red_table()
        elif self.output_mode == "vector":
            obs = self._create_red_vector()
        elif self.output_mode == "raw":
            obs = observation
        else:
            raise NotImplementedError("Invalid output_mode")
    
        return obs
    
    def _initial_red_obs(self, obs):
        for hostid in obs:
            if hostid == "success":
                continue
            host = obs[hostid]
            interface = host["Interface"][0]
            subnet = interface["Subnet"]
            self.known_subnets.add(subnet)
            ip = str(interface["IP Address"])
            hostname = host["System info"]["Hostname"]
            self.red_info[ip] = [str(subnet), str(ip), hostname, False, "Privileged"]

    def _update_red_info(self, obs, action):
        name = action.__class__.__name__
        if name == "DiscoverRemoteSystems":
            self._add_ips(obs)
        elif name == "DiscoverNetworkServices":
            ip = str(action.ip_address)
            if (ip in self.red_info):
                self.red_info[ip][3] = True
        elif name == "ExploitRemoteService":
            self._process_exploit(obs)
        elif name == "PrivilegeEscalate":
            hostname = str(action.hostname)
            if(str(self.ip_map[hostname]) in self.red_info):
                self._process_priv_esc(obs, hostname)
        elif name == "Impact":
            hostname = str(action.hostname)
            if hostname == "Op_Server0":
                server_ip = str(self.ip_map[hostname])
                if(server_ip in self.red_info):
                    access = self.red_info[server_ip][4]
                    if access == "Privileged":
                        self.impact_status = [1]

    
    def _add_ips(self, obs):
        for hostid in obs:
            if hostid == "success":
                continue
            host = obs[hostid]
            for interface in host["Interface"]:
                ip = interface["IP Address"]
                subnet = interface["Subnet"]
                if subnet not in self.known_subnets:
                    self.known_subnets.add(subnet)
                if str(ip) not in self.red_info:
                    subnet = self._get_subnet(ip)
                    hostname = self._generate_name("HOST")
                    self.red_info[str(ip)] = [subnet, str(ip), hostname, False, "None"]
                # elif self.red_info[str(ip)][0].startswith("UNKNOWN_"):
                # elif "UNKNOWN_" in self.red_info[str(ip)][0]:
                else:
                    self.red_info[str(ip)][0] = self._get_subnet(ip)
    
    def _get_subnet(self, ip):
        for subnet in self.known_subnets:
            if ip in subnet:
                return str(subnet)
        return self._generate_name("SUBNET")
    
    def _generate_name(self, datatype: str):
        self.id_tracker += 1
        unique_id = "UNKNOWN_" + datatype + ": " + str(self.id_tracker)
        return unique_id

    def _process_exploit(self, obs):
        for hostid in obs:
            if hostid == "success":
                continue
            host = obs[hostid]
            if "Sessions" in host:         
                ip = str(host["Interface"][0]["IP Address"])
                hostname = host["System info"]["Hostname"]
                session = host["Sessions"][0]

                # if Red already has Root access, it keeps this access and does not drop to User
                if self.red_info[ip][4] == "Privileged":
                    access = "Privileged"
                else:
                    # access = "Privileged" if "Username" in session else "User" # this needs to be changed in original CybORG code. Privileged means the username is SYSTEM or root
                    # this access is true when all machines use windows_user_host1 image (which has 'vagrant' username on a regular exploit, no username on a critical exploit,
                    # and 'SYSTEM' username on a privilege escalation)
                    # only the OpServer uses its own image (which has 'pi' username on a regular exploit, and 'root' username on a privilege escalation)
                    access = "User"
                    if "Username" in session:
                        if session['Username'] in {'root','SYSTEM'}:
                            access = "Privileged"
                    else:
                        if hostname != "Op_Host0":
                            access = "Privileged"
                        
                self.red_info[ip][2] = hostname
                self.red_info[ip][4] = access
    
    def _process_priv_esc(self, obs, hostname):
        if obs["success"] == False:
            self.red_info[str(self.ip_map[hostname])][4] = 'None'

        else:
            for hostid in obs:
                if hostid == "success":
                    continue
                host = obs[hostid]
                ip = host["Interface"][0]["IP Address"]
                if "Sessions" in host:
                    
                    access = "Privileged"
                    self.red_info[str(ip)][4] = access
                    # New: learn subnet with Root access to a host 
                    subnet = host["Interface"][0]["Subnet"]
                    if subnet not in self.known_subnets:
                        self.known_subnets.add(subnet)
                        self.red_info[str(ip)][0] = subnet
                else:
                    subnet = self._get_subnet(ip) # this line is replaced above
                    hostname = self._generate_name("HOST")

                    if str(ip) not in self.red_info:
                        self.red_info[str(ip)] = [
                            subnet,
                            str(ip),
                            hostname,
                            False,
                            "None",
                        ]
                    else:
                        self.red_info[str(ip)][0] = subnet
                        self.red_info[str(ip)][2] = hostname
    
    def _create_red_vector(self):
        table = self._create_red_table()._rows

        # success flag for previous action. Not included for now
        # success = int(self.success.value) if self.success.value < 2 else 0

        # Four flags for each subnet:
        # s1 - are the IPs still unknown
        # s2 - are the hosts still not scanned
        # s3 - still no User access on any machine
        # s4 - still no Root access on any machine

        # Four flags for each host:
        # h1 - has this host been scanned
        # h2/h3/h4 - is there None, User, or Privileged access on this host

        # one bit to track if Red is currently impacting

        subnet_status = [0]*2*len(self.subnets)
        host_status = [1,0,0]*len(self.hostnames)
        
        for host in self.hostnames:
            if str(self.ip_map[host]) in self.red_info:
                host_ip = str(self.ip_map[host])
                index = self.hostnames.index(host)

                # s1, are the IPs in this subnet known
                if self.subnet_scan_status[0] == 0:
                    if (str(self.red_info[host_ip][0]) == str(self.subnet_map["User"])):
                        self.subnet_scan_status[0] = 1
                elif self.subnet_scan_status[1] == 0:
                    if (str(self.red_info[host_ip][0]) != str(self.subnet_map["User"])):
                        self.subnet_scan_status[1] = 1
                    
                # h1 - has this host been scanned
                if int(self.red_info[host_ip][3]):
                    self.host_scan_status[index] = 1
                    # update s2
                    if (str(self.red_info[host_ip][0]) == str(self.subnet_map["User"])):
                        self.subnet_scan_status[2] = 1
                    elif (str(self.red_info[host_ip][0]) != str(self.subnet_map["User"])):
                        self.subnet_scan_status[3] = 1

                # h2/h3/h4 - is there none/user/root access on this host
                access = self.red_info[host_ip][4]
                if access == "User":
                    host_status[(3*index)] = 0
                    host_status[(3*index)+1] = 1
                    host_status[(3*index)+2] = 0
                    # update s3
                    if (str(self.red_info[host_ip][0]) == str(self.subnet_map["User"])):
                        subnet_status[0] = 1
                    elif (str(self.red_info[host_ip][0]) != str(self.subnet_map["User"])):
                        subnet_status[1] = 1

                elif access == "Privileged":
                    host_status[(3*index)] = 0
                    host_status[(3*index)+1] = 0
                    host_status[(3*index)+2] = 1
                    # update s4
                    if (str(self.red_info[host_ip][0]) == str(self.subnet_map["User"])):
                        subnet_status[2] = 1
                    elif (str(self.red_info[host_ip][0]) != str(self.subnet_map["User"])):
                        subnet_status[3] = 1

                if host == "Op_Server0":
                    if access != "Privileged":    
                        self.impact_status = [0]

        proto_vector = []
        proto_vector.extend(self.host_scan_status) # 5 bits
        proto_vector.extend(host_status) # 15 bits
        proto_vector.extend(self.subnet_scan_status) # 4 bits
        proto_vector.extend(subnet_status) # 4 bits
        proto_vector.extend(self.impact_status) # 1 bit
        turn_vector = [0]*(self.turns_per_game+1)
        turn_vector[self.turn] = 1
        proto_vector.extend(turn_vector)

        return np.array(proto_vector)
    
    def _create_red_table(self):
        # The table data is all stored inside the ip nodes
        # which form the rows of the table
        table = PrettyTable(
            [
                "Subnet",
                "IP Address",
                "Hostname",
                "Scanned",
                "Access",
            ]
        )
        for ip in self.red_info:
            table.add_row(self.red_info[ip])

        table.sortby = "IP Address"
        table.success = self.success
        return table
    
    def get_attr(self, attribute: str):
        return self.env.get_attr(attribute)

    def get_observation(self, agent: str):
        if agent == "Blue" and self.output_mode == "table":
            output = self.get_table()
        else:
            output = self.get_attr("get_observation")(agent)

        return output

    def get_agent_state(self, agent: str):
        return self.get_attr("get_agent_state")(agent)

    def get_action_space(self, agent):
        return self.env.get_action_space(agent)

    def get_last_action(self, agent):
        return self.get_attr("get_last_action")(agent)

    def get_ip_map(self):
        return self.get_attr("get_ip_map")()

    def get_rewards(self):
        return self.get_attr("get_rewards")()


