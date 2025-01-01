import networkx as nx
import matplotlib.pyplot as plt


line1 = [
    "Академмістечко", "Житомирська", "Святошин", "Нивки",
    "Берестейська", "Шулявська", "Політехнічний інститут", "Вокзальна",
    "Університет", "Театральна", "Хрещатик", "Арсенальна",
    "Дніпро", "Гідропарк", "Лівобережна", "Дарниця",
    "Чернігівська", "Лісова"
]

line2 = [
    "Героїв Дніпра", "Мінська", "Оболонь", "Почайна",
    "Тараса Шевченка", "Контрактова площа", "Поштова площа", "Майдан Незалежності",
    "Площа Льва Толстого", "Олімпійська", "Палац Україна", "Либідська",
    "Деміївська", "Голосіївська", "Васильківська", "Виставковий центр",
    "Іподром", "Теремки"
]

line3 = [
    "Сирець", "Дорогожичі", "Лук'янівська", "Золоті Ворота",
    "Палац спорту", "Кловська", "Печерська", "Дружби народів",
    "Видубичі", "Славутич", "Осокорки", "Позняки",
    "Харківська", "Вирлиця", "Бориспільська", "Червоний хутір"
]

interchanges = [
    ("Театральна", "Золоті Ворота"),
    ("Хрещатик", "Майдан Незалежності"),
    ("Площа Льва Толстого", "Палац спорту")
]

G = nx.Graph()

# Додаємо вершини та ребра для Лінії 1
for i in range(len(line1) - 1):
    G.add_node(line1[i])
    G.add_node(line1[i+1])
    G.add_edge(line1[i], line1[i+1])

# Додаємо вершини та ребра для Лінії 2
for i in range(len(line2) - 1):
    G.add_node(line2[i])
    G.add_node(line2[i+1])
    G.add_edge(line2[i], line2[i+1])

# Додаємо вершини та ребра для Лінії 3
for i in range(len(line3) - 1):
    G.add_node(line3[i])
    G.add_node(line3[i+1])
    G.add_edge(line3[i], line3[i+1])

# Додаємо пересадкові вузли
for s1, s2 in interchanges:
    G.add_node(s1)
    G.add_node(s2)
    G.add_edge(s1, s2)

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

print("Кількість станцій (вершин) у графі:", num_nodes)
print("Кількість зв'язків (ребер) у графі:", num_edges)

degrees = dict(G.degree())
print("Ступені кожної вершини:")
for station, deg in degrees.items():
    print(f"{station}: {deg}")

plt.figure(figsize=(12, 8))

pos = nx.spring_layout(G, k=0.7, seed=42)  # seed для фіксованого розміщення

# Відмальовуємо ребра
nx.draw_networkx_edges(G, pos, edge_color="gray", width=2)

# Відмальовуємо вершини
nx.draw_networkx_nodes(G, pos, node_size=500, node_color="lightblue")

# Підписи до вершин
nx.draw_networkx_labels(G, pos, font_size=9)

plt.title("Спрощена модель Київського метрополітену у вигляді графа", fontsize=14)
plt.axis("off")
plt.show()

from collections import deque

def bfs_path(graph, start, goal):
    """
    Знаходить (один) найкоротший шлях від вершини start до goal
    у невагомому графі за допомогою пошуку в ширину (BFS).
    """
    visited = set()            # множина відвіданих вершин
    queue = deque([[start]])   # черга, яка зберігає шляхи (на початку шук. шлях = [start])

    while queue:
        path = queue.popleft()     # дістати перший шлях із черги
        node = path[-1]           # остання вершина в поточному шляху

        if node == goal:
            return path  # Якщо ми досягли цільової вершини, повертаємо знайдений шлях

        if node not in visited:
            visited.add(node)
            # Додаємо в чергу нові шляхи, що містять сусідів node
            for neighbor in graph[node]:
                if neighbor not in visited:
                    new_path = list(path)  # копія поточного шляху
                    new_path.append(neighbor)
                    queue.append(new_path)
    return None

def dfs_path(graph, start, goal):
    """
    Знаходить (один) шлях від вершини start до goal
    за допомогою пошуку в глибину (DFS).
    Не обов'язково є найкоротшим.
    """
    visited = set()
    stack = [[start]]  # стос (stack), де кожен елемент — це шлях

    while stack:
        path = stack.pop()     # беремо останній шлях
        node = path[-1]        # остання вершина

        if node == goal:
            return path  # Якщо знайшли ціль, повертаємо поточний шлях

        if node not in visited:
            visited.add(node)
            # Додаємо в stack усіх сусідів (шлях + сусід)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    new_path = list(path)
                    new_path.append(neighbor)
                    stack.append(new_path)
    return None

start_station = "Академмістечко"
goal_station = "Лівобережна"

path_bfs = bfs_path(G, start_station, goal_station)
path_dfs = dfs_path(G, start_station, goal_station)

print("BFS-шлях (найкоротший за кількістю станцій):")
print(" -> ".join(path_bfs))

print("\nDFS-шлях (перший-ліпший 'глибокий' шлях):")
print(" -> ".join(path_dfs))

import random

G_weighted = nx.Graph()

def add_weighted_edges(line):
    for i in range(len(line) - 1):
        # Додаємо ребро з випадковою вагою в інтервалі [1..10]
        w = random.randint(1, 10)
        G_weighted.add_edge(line[i], line[i+1], weight=w)

# Додаємо лінії з вагою
add_weighted_edges(line1)
add_weighted_edges(line2)
add_weighted_edges(line3)

# Додаємо пересадки (міжлінійні) переходи теж із випадковою вагою
for s1, s2 in interchanges:
    w = random.randint(1, 10)
    G_weighted.add_edge(s1, s2, weight=w)

import heapq

def dijkstra(graph, start):
    """
    Повертає словник {vertex: distance}, де distance – найкоротша відстань від start до vertex,
    обчислена за алгоритмом Дейкстри (з урахуванням ваг edge['weight']).
    """
    # Крок 1: ініціалізація відстаней
    distances = {node: float('inf') for node in graph.nodes}
    distances[start] = 0

    # Пріоритетна черга (відкриті вершини): (distance, node)
    pq = [(0, start)]  # відстань 0 для старту

    # Поки черга не порожня:
    while pq:
        current_dist, current_vertex = heapq.heappop(pq)

        # Якщо витягнута відстань більша за вже записану, пропускаємо (бо це "застаріла" інформація)
        if current_dist > distances[current_vertex]:
            continue

        # Крок 2: "розслаблення" (relax) сусідів
        for neighbor in graph[current_vertex]:
            edge_weight = graph[current_vertex][neighbor]['weight']
            distance = current_dist + edge_weight

            # Якщо знайшли коротший шлях до сусіда:
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances

all_pairs_shortest_paths = {}

# Ітеруємось за всіма вершинами як стартовими
for start_node in G_weighted.nodes():
    distances = dijkstra(G_weighted, start_node)
    all_pairs_shortest_paths[start_node] = distances

# Приклад виводу: відстані від "Академмістечко" до інших
print("Найкоротші відстані від станції 'Академмістечко':")
for station, dist in all_pairs_shortest_paths["Академмістечко"].items():
    print(f"{station:25s} -> {dist}")

def dijkstra_with_path(graph, start):
    distances = {node: float('inf') for node in graph.nodes}
    distances[start] = 0
    predecessors = {node: None for node in graph.nodes}

    pq = [(0, start)]
    while pq:
        current_dist, current_vertex = heapq.heappop(pq)
        if current_dist > distances[current_vertex]:
            continue

        for neighbor in graph[current_vertex]:
            edge_weight = graph[current_vertex][neighbor]['weight']
            distance = current_dist + edge_weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_vertex
                heapq.heappush(pq, (distance, neighbor))

    return distances, predecessors
