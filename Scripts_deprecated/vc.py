import time
from hbp_nrp_virtual_coach.virtual_coach import VirtualCoach
vc = VirtualCoach(environment='local', storage_username='nrpuser', storage_password='password')
vc.print_cloned_experiments()
vc.print_available_servers()
vc.print_running_experiments()

sim = vc.launch_experiment('dataset_generator_0')
logMe("Starting NRP Simulation")
#sim.start()
#time.sleep(10000)
#sim.stop()


