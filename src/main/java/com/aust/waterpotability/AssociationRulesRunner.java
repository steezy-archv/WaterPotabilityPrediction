package com.aust.waterpotability;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.associations.Apriori;
import weka.associations.FPGrowth;
import weka.associations.AssociationRules;
import weka.associations.AssociationRule;

public class AssociationRulesRunner {

    public static void main(String[] args) throws Exception {

        DataSource source = new DataSource("data/raw_processed_water_potability.arff");
        Instances data = source.getDataSet();

        Discretize discretize = new Discretize();
        discretize.setInputFormat(data);
        Instances newData = Filter.useFilter(data, discretize);
        System.out.println("Discretized numeric attributes for association rule mining.");

        // === Choose algorithm ===
        String algorithm = "Apriori"; // change to "Apriori" or "FP"

        if (algorithm.equalsIgnoreCase("Apriori")) {
            System.out.println("=== Running Apriori ===");
            Apriori apriori = new Apriori();
            apriori.setClassIndex(newData.classIndex()); 
            apriori.setLowerBoundMinSupport(0.05); 
            apriori.setMinMetric(0.6);   
            apriori.buildAssociations(newData);

            System.out.println(apriori);
        } else if (algorithm.equalsIgnoreCase("FP")) {
            System.out.println("=== Running FP-Growth ===");
            FPGrowth fpGrowth = new FPGrowth();
            fpGrowth.setNumRulesToFind(10);
            fpGrowth.buildAssociations(newData);

            AssociationRules rules = fpGrowth.getAssociationRules();
            for (AssociationRule rule : rules.getRules()) {
                System.out.println(rule);
            }
        } else {
            System.out.println("Invalid algorithm choice! Use 'Apriori' or 'FP'.");
        }
    }
}

