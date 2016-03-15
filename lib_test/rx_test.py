import rx
from rx import Observable, Observer
from itertools import combinations
from rx.subjects import Subject

#create class that generates 100 random values and X subscribers that print it
#create on "larger" observer that collects all emitted data

#root is a subject

root = Subject()

n_children=5
n_samples=20
#first subscribe all it's children, filter out the amount of children

#merge all leaf children(no children of their own) in a single observer to check when completed

leaves=Observable.empty()
for i in range(7):
    child=root.filter(
                        lambda var, i: i<var[1]
                    ).map(
                        lambda var, i: var[0] * 2
                    )
    leaves=leaves.merge(child)
class NodeObserver(Observer):
    def on_next(self, parent_node, i):
        print(parent_node,i)

    def on_error(self, e):
        print("Got error: %s" % e)

    def on_completed(self):
        print("Sequence completed")
#wait for oncomplete or error
t=[]
test = root.subscribe(t.append)
#leaves.ignore_elements().subscribe(End_Of_Sampling(rc))
#when sampling is called the values are calculated
for i in range(n_samples):
    #pass parent sample in onnext
    root.on_next((i,n_children))

#do the training in this subscription
leaves.ignore_elements().subscribe(print(t))