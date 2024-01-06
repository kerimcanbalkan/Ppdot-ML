<h1>Pulsar Type Classification using Random Forest</h1>
<h2>Overview</h2>
<p>This project aims to classify pulsars into different types based on their characteristic features, specifically the
pulse period (P0), period derivative (P1), and surface magnetic field (BSURF). The classification is performed
using the Random Forest algorithm, a machine learning technique.</p>

<h2>Dataset</h2>

<p>The dataset used in this project is obtained from the ATNF Pulsar Catalogue, containing information about pulsars'
P0, P1, TYPE, and BSURF parameters.</p>

<h2>Requirements</h2>
<ul>
        <li>Python 3.x</li>
        <li>pandas</li>
        <li>numpy</li>
        <li>matplotlib</li>
        <li>scikit-learn</li>
</ul>

<h2>Usage</h2>

<ol>
        <li>Install the required dependencies using:
            <code>pip install pandas numpy matplotlib scikit-learn</code></li>
        <li>Run the project file:
            <code>python main.py</code></li>
        <li>The Random Forest classifier will be trained on the data, and predictions will be made on a test set.</li>
        <li>Accuracy and additional classification metrics will be displayed.</li>
        <li>Three plots will be generated and saved:
            <ul>
                <li>Original Data</li>
                <li>Test Data</li>
                <li>Predicted Data</li>
            </ul>
        </li>
</ol>

<h2>Results</h2>

<p>The Random Forest classifier achieves an accuracy of approximately 78%, indicating its effectiveness in classifying
        pulsar types based on the provided features.</p>

<h2>File Descriptions</h2>

<ul>
        <li><code>your_project_file.py</code>: Main script containing the implementation of the Random Forest
            classifier.</li>
        <li><code>data/data1.csv</code>: CSV file containing the ATNF Pulsar Catalogue data.</li>
        <li><code>README.md</code>: This file providing an overview of the project.</li>
</ul>

<h2>Acknowledgments</h2>

<p>ATNF Pulsar Catalogue: <a href="http://www.atnf.csiro.au/people/pulsar/psrcat/">Link to ATNF Pulsar
            Catalogue</a></p>
