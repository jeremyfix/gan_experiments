set key autotitle columnhead
set datafile separator comma

set terminal epslatex standalone
set output "metrics.tex"

set multiplot layout 3, 1 title "DCGAN on MNIST"
unset key

set title "Generator loss $-\\frac{1}{B}\\sum_i \\log(D(G(z_i)))$"
set ytics 0,4,8
set yrange [0:8]
plot "generator_loss.csv" using 2:3 with lines

set title "Critic loss $-\\frac{1}{B}\\sum_i (\\log(D(x_i)) + \\log(1-D(G(z_i))))$"
set ytics 0,0.2,0.4
set yrange [0:0.4]
plot "critic_loss.csv" using 2:3 with lines

set title "Critic accuracy"
set ytics 0,0.5,1
set yrange [0:1.1]
plot "critic_accuracy.csv" using 2:3 with lines

unset multiplot
