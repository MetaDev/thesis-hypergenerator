from pomegranate import *


c = MixtureDistribution( [ NormalDistribution( 2, 4 ), NormalDistribution( 2, 4 ) ], weights=[1, 1] )

child0=State(c,"child0")
child1=State(c,"child1")
hmm = HiddenMarkovModel( "HT" )
hmm.add_states([child0, child1])

hmm.add_transition( hmm.start, child0, 1 )
hmm.add_transition( child0, child1, 0.5 )

hmm.add_transition( child1, hmm.end, 1 )
hmm.add_transition( child0, hmm.end, 0.5 )
hmm.bake()

sequence = hmm.sample()
print( sequence)