package com.aust.waterpotability;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.VisualizePanel;

import javax.swing.*;

public class UnsupervisedModel {
    public static void main(String[] args) throws Exception {

        DataSource source = new DataSource("data/processed_water_potability.arff");
        Instances data = source.getDataSet();

        // Remove the Potability class attribute for clustering
        Remove remove = new Remove();
        remove.setAttributeIndices("" + (data.numAttributes())); // remove last attribute
        remove.setInputFormat(data);
        Instances clusterData = Filter.useFilter(data, remove);

        // Build KMeans with 2 clusters
        SimpleKMeans kmeans = new SimpleKMeans();
        kmeans.setNumClusters(2);
        kmeans.buildClusterer(clusterData);

        // Evaluate clusters
        ClusterEvaluation eval = new ClusterEvaluation();
        eval.setClusterer(kmeans);
        eval.evaluateClusterer(clusterData);

        System.out.println("Cluster Evaluation:");
        System.out.println(eval.clusterResultsToString());

        // Add cluster assignments back to dataset
        Instances labeled = new Instances(clusterData);
        labeled.insertAttributeAt(new weka.core.Attribute("Cluster"), labeled.numAttributes());
        for (int i = 0; i < clusterData.numInstances(); i++) {
            int cluster = kmeans.clusterInstance(clusterData.instance(i));
            labeled.instance(i).setValue(labeled.numAttributes() - 1, cluster);
        }

        // Prepare plot
        PlotData2D plotData = new PlotData2D(labeled);
        plotData.setPlotName("KMeans Clustering");
        plotData.addInstanceNumberAttribute();

        VisualizePanel vp = new VisualizePanel();
        vp.setName("Cluster Visualization");
        vp.addPlot(plotData);

        // Display in a JFrame
        JFrame jf = new JFrame("Cluster Visualization");
        jf.setSize(800, 600);
        jf.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        jf.getContentPane().setLayout(new java.awt.BorderLayout());
        jf.getContentPane().add(vp, java.awt.BorderLayout.CENTER);
        jf.setVisible(true);
    }
}

