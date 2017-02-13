from random import randint

class Node(object):
    n_actions = 0

    def __init__(self, parent=None):
        self.parent = parent
        self.childs = []


def add_nodes(node):
    for i in xrange(10):
        new_node = Node(node)
        node.childs.append(new_node)


def build_tree(node, depth):
    d = 0
    while d < depth:
        add_nodes(node)
        node = node.childs[randint(0, len(node.childs) - 1)]
        d += 1


def main():
    Node.n_actions = 10
    node = Node()

    for _ in xrange(10000000):
        build_tree(node, 100)

        node = node.childs[3]

if __name__ == '__main__':
    main()
