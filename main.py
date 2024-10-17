import yaml
from sys import argv
from enum import Enum

class EventType(Enum):
    ARRIVE = 'arrive'
    EXIT = 'exit'
    MOVE = 'move'

class Event:
    def __init__(self, type, time, source, target) -> None:
        self.type = type
        self.time = time
        self.source = source
        self.target = target

    def __str__(self):
        if self.source == None:
            return f'Type:{self.type} | Time: {self.time} | Source: {self.source} | Target: {self.target.id}'
        if self.target == None:
            return f'Type:{self.type} | Time: {self.time} | Source: {self.source.id} | Target: {self.target}'
        return f'Type:{self.type} | Time: {self.time} | Source: {self.source.id} | Target: {self.target.id}'

class PseudoRandomNumbers:
    def __init__(self, seed, total_numbers, random_numbers = None, generate=False) -> None:
        self.m = 2**28
        self.a = 1317293
        self.c = 12309820398
        self.seed = seed

        self.x = seed
        self.total_numbers = total_numbers

        self.numbers = random_numbers
        self.generate = generate

        if generate == False:
            self.total_numbers = len(random_numbers)

        self.current = -1

    def gen_rand(self, n):
        x = self.seed #seed
        arr = []
        for _ in range(n):
            if _ % 10000000 == 0: print(_)
            op = (self.a * x + self.c) % self.m
            x = op
            arr.append(op/self.m)
        return arr
    
    def get_next_number(self):
        self.current += 1
        if self.numbers and not self.generate:
            return self.numbers[self.current % self.total_numbers]
        op = (self.a * self.x + self.c) % self.m
        self.x = op
        return op/self.m
    
    def reset(self):
        self.x = self.seed
        self.current = -1

class Scheduler():
    def __init__(self, random_numbers: PseudoRandomNumbers):
        self.random_numbers = random_numbers
        self.scheduler = []

    def add(self, event, interval):
        if self.random_numbers.current == self.random_numbers.total_numbers:
            return
        event.time = event.time + self.get_random(interval)
        self.scheduler.append(event)
        self.scheduler.sort(key=lambda event: event.time)

    def add_rand(self, event, rand_num):
        event.time = event.time + rand_num
        self.scheduler.append(event)
        self.scheduler.sort(key=lambda event: event.time)

    def schedule(self) -> Event:
        return self.scheduler.pop(0)
    
    def get_random(self, interval) -> float:
        rand_num = self.random_numbers.get_next_number()
        return interval.start + (interval.end - interval.start) * rand_num

class Queue:
    def __init__(self, id, capacity, servers, arrival_interval, service_interval) -> None:
        self.id = id
        self.capacity = capacity 
        self.servers = servers
        self.arrival_interval = arrival_interval
        self.service_interval = service_interval
        self.status = 0
        self.losses = 0
        self.states = [0] * (capacity + 1)
        self.queues_candidate = [] # array (queue, prob) 

    def add(self):
        self.status = self.status + 1

    def out(self):
        self.status = self.status - 1

    def loss(self):
        self.losses = self.losses + 1

    def update_states(self, time):
        self.states[self.status] = self.states[self.status] + time 
        
    def add_queue(self, queue, prob):
        self.queues_candidate.append((queue, prob))
        
    def target(self, prob, time):
        threshold = 1e-6  # or any other suitable threshold
        cumulative_prob = 0
        for target, queue_prob in self.queues_candidate:
            cumulative_prob += queue_prob
            if prob < cumulative_prob + threshold:
                return Event(EventType.MOVE, time, self, target)
        return Event(EventType.EXIT, time, self, None)
    
    def __str__(self) -> str:
        string = f'Capacity: {self.capacity}' + '\n'
        string = string + f'Servers: {self.servers}' + '\n'
        string = string + f'Arrival Interval: {self.arrival_interval}' + '\n'
        string = string + f'Service Interval: {self.service_interval}' + '\n'
        string = string + f'Status: {self.status}' + '\n'
        string = string + f'Losses: {self.losses}' + '\n'
        string = string + f'States: {self.states}'

        return string

class Interval:
    def __init__(self, start, end) -> None:
        self.start = start
        self.end = end

    def __str__(self) -> str:
        return f'Start {self.start} | End {self.end}'

class Simulation:
    def __init__(self, arrival_time, queues, scheduler):
        self.arrival_time = arrival_time
        self.queues = queues
        self.scheduler = scheduler
        self.global_time = 0

    def run(self):
        first_queue = self.queues[0]
        self.scheduler.add_rand(Event(EventType.ARRIVE, self.arrival_time, None, first_queue), 0)
        while (self.scheduler.random_numbers.current + 2) <= self.scheduler.random_numbers.total_numbers: # tem que colocar +2 pq o current comeca com -1 
            next_event = self.scheduler.schedule()
            
            self.__update_global_time(next_event)

            if (next_event.type == EventType.ARRIVE):
                self.arrival(None, next_event.target)
            elif (next_event.type == EventType.EXIT):
                self.exit(next_event.source, None)
            elif (next_event.type == EventType.MOVE):
                self.move(next_event.source, next_event.target)
        
    def arrival(self, _, target: Queue):
        if target.status < target.capacity:
            target.add()
            if target.status <= target.servers:
                event = target.target(self.scheduler.get_random(Interval(0, 1)), self.global_time)
                self.scheduler.add(event, target.service_interval)
        else:
            target.loss()
        self.scheduler.add(Event(EventType.ARRIVE, self.global_time, None, target), target.arrival_interval)

    def exit(self, source, _):
        source.out()
        if source.status >= source.servers:
            self.scheduler.add(source.target(self.scheduler.get_random(Interval(0, 1)), self.global_time), source.service_interval)
            
    def move(self, source, target):
        source.out()
        if source.status >= source.servers:
            self.scheduler.add(source.target(self.scheduler.get_random(Interval(0, 1)), self.global_time), source.service_interval)
        if target.status < target.capacity:
            target.add()
            if target.status <= target.servers:
                self.scheduler.add(target.target(self.scheduler.get_random(Interval(0, 1)), self.global_time), target.service_interval)
        else:
            target.loss()

    def __update_global_time(self, event):
        for queue in self.queues:
            queue.update_states(event.time - self.global_time)
        self.global_time = event.time

class Stats:
    def __init__(self, simulation):
        self.simulation = simulation 

    def calc_prob_distribution(self, queue):
        distribution = [0] * (queue.capacity + 1)
        states = queue.states
        global_time = self.simulation.global_time

        for index, state in enumerate(states):
            distribution[index] = (index, state, state/global_time)

        return distribution

    def show_prob_distribution(self, queue):
        distribution = self.calc_prob_distribution(queue)

        print("State\t\tTime\t\tProbability")
        for row in distribution:
            if row[1] != 0:
                print(f"{row[0]}\t\t{round(row[1], 4)}\t\t{row[2] * 100:,.2f}%")

    def show_global_time(self):
        print("Simulation average time:", self.simulation.global_time)

    def show_losses(self, queue):
        print("Number of losses:", queue.losses)
    
    def report(self):
        for index, queue in enumerate(self.simulation.queues): 
            k = queue.capacity
            print("***********************************************************")
            if not k == 100:
                print(f"Queue:   Q{index+1} (G/G/{queue.servers}/{queue.capacity})")
            else:
                print(f"Queue:   Q{index+1} (G/G/{queue.servers})")
            if queue.arrival_interval != None:
                print(f"Arrival: {queue.arrival_interval.start} ... {queue.arrival_interval.end}")
            print(f"Service: {queue.service_interval.start} ... {queue.service_interval.end}")
            print("***********************************************************")
            self.show_prob_distribution(queue)
            self.show_losses(queue)

        self.show_global_time()

def load_config(file_name):
    with open(file_name) as stream:
        try:
            CONFIG = yaml.safe_load(stream)
        except:
            print('=== ERROR LOADING YAML FILE ===')
            exit(0)

    return CONFIG

def get_queues(config) -> list:
    queues_config = config['queues']

    queues = []

    for _, queue_id in enumerate(queues_config):
        queue_config = queues_config[queue_id]

        servers = queue_config['servers']
        service_interval = Interval(queue_config['minService'], queue_config['maxService'])

        if "capacity" in queue_config:
            capacity = queue_config['capacity']
        else:
            capacity = 100 ## preguica de fazer um jeito melhor, mas o certo seria colocar um valor -1 aqui e tratar nos outros lugares do codigo

        if "minArrival" in queue_config:
            arrival_interval = Interval(queue_config['minArrival'], queue_config['maxArrival'])
        else:
            arrival_interval = None
            
        queues.append(Queue(capacity=capacity, id=queue_id, servers=servers, arrival_interval=arrival_interval, service_interval=service_interval))
    
    return queues

# not used
def get_backbone(config, queue_size) -> list:
    network = config["network"]
    backbone = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]

    for event in network:
        source = event["source"]
        target = event["target"]
        probability = event["probability"]
        s = int(source[1:])
        t = int(target[1:])
        backbone[s][t] = float(probability)

    # verifica a probabilidade para fila de saida
    for queue_events in backbone:
        sum_prob = sum(queue_events)
        if sum_prob != 1:
            queue_events[0] = round(1 - sum_prob, 1)

    return backbone

def add_network(source_id, target_id, prob, queues: list):
    source: Queue = queues[int(source_id[1:]) - 1]
    target: Queue = queues[int(target_id[1:]) - 1]
    source.add_queue(target, prob)

def main():
    CONFIG = load_config(argv[1])

    arrival_time = CONFIG['arrivals']['Q1']

    seeds = CONFIG['seed']
    
    queues = get_queues(CONFIG)
    
    network = CONFIG["network"]
    
    for event in network:
        add_network(event["source"], event["target"], event["probability"], queues)
        
    total_rnd_numbers = CONFIG['rndnumbersPerSeed']

    random_numbers = CONFIG.get('rndnumbers')
      
    random_numbers = PseudoRandomNumbers(seeds[0], total_rnd_numbers, random_numbers=random_numbers, generate=not bool(random_numbers))
    
    scheduler = Scheduler(random_numbers)

    sim = Simulation(arrival_time=arrival_time, queues=queues, scheduler=scheduler)

    sim.run()

    Stats(sim).report()

if __name__ == '__main__':
    main()