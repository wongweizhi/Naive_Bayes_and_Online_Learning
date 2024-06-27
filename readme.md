# Naive Baye's Image Classifier and Online Learning

<ol>
    <li>Create a Naive Baye's Image Classifier that support <b>discrete</b> and <b>continuous</b> features.</li>
    <li>Use online learning to get the parameter of beta distribution.</li>
</ol>

## 1. Naive Baye's Image Classifier
Create a Naive Baye's Image Classifier that support <b>discrete</b> and <b>continuous</b> features.

$$
P(class|data) = \frac{P(data|claass)P(class)}{P(data)}
$$

<ul>
    <li>Dataset: MNIST</li>
    <li>Hyperparameter:</li>
        <ol>
            <li>NumofClass: the number of class, in this case, that is 10 (0 ~ 9)
            <li>Mode: there are discrete and continuous mode</li>
            <li>NumofBin: the number of bin which split the color level 256. It is only needed in discrete mode.
        </ol>
    <li>Output: it would print each class posterior probability of every test data and also the error rate.</li>
    <li>You can also use  the function <b>prob_visualize</b> to have a visulaizaiton of the posteror.</li>
    
</ul>

## 2.Online Learning

Use online learning to get the parameter of beta distribution.

$$
pdf: f(x, \alpha, \beta)=\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}x^{\alpha-1}(1-x)^{\beta-1},  \\
$$
$$
\Gamma(z+1) = z\Gamma(z), \Gamma(1)=1
$$ 
<ul>
    <li>Input: the txt file contain 0 and 1</li>
    <li>Hyperparameter: inital alpha and beta</li>
    <li>Output: print each line in the file, prior, likelihood, and posterior</li>
</ul>
