python multiarmed_bandits.py --mode=greedy --epsilon=0.25
python multiarmed_bandits.py --mode=greedy --epsilon=0.125
python multiarmed_bandits.py --mode=greedy --epsilon=0.0625
python multiarmed_bandits.py --mode=greedy --epsilon=0.03125
python multiarmed_bandits.py --mode=greedy --epsilon=0.015625

python multiarmed_bandits.py --mode=greedy --alpha=0.15 --epsilon=0.25
python multiarmed_bandits.py --mode=greedy --alpha=0.15 --epsilon=0.125
python multiarmed_bandits.py --mode=greedy --alpha=0.15 --epsilon=0.0625
python multiarmed_bandits.py --mode=greedy --alpha=0.15 --epsilon=0.03125
python multiarmed_bandits.py --mode=greedy --alpha=0.15 --epsilon=0.015625

python multiarmed_bandits.py --mode=greedy --alpha=0.15 --initial=1 --epsilon=0.0625
python multiarmed_bandits.py --mode=greedy --alpha=0.15 --initial=1 --epsilon=0.03125
python multiarmed_bandits.py --mode=greedy --alpha=0.15 --initial=1 --epsilon=0.015625
python multiarmed_bandits.py --mode=greedy --alpha=0.15 --initial=1 --epsilon=0.0078125

python multiarmed_bandits.py --mode=ucb --c=0.25
python multiarmed_bandits.py --mode=ucb --c=0.5
python multiarmed_bandits.py --mode=ucb --c=1
python multiarmed_bandits.py --mode=ucb --c=2
python multiarmed_bandits.py --mode=ucb --c=4

python multiarmed_bandits.py --mode=gradient --alpha=0.0625
python multiarmed_bandits.py --mode=gradient --alpha=0.125
python multiarmed_bandits.py --mode=gradient --alpha=0.25
python multiarmed_bandits.py --mode=gradient --alpha=0.5
