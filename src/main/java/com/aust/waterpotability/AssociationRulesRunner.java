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

        DataSource source = new DataSource("data/processed_water_potability.arff");
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
            apriori.setClassIndex(newData.classIndex()); // optional
            apriori.setLowerBoundMinSupport(0.1); // support threshold
            apriori.setMinMetric(0.7);            // confidence threshold
            apriori.buildAssociations(newData);

            // Print rules
            System.out.println(apriori);
        } else if (algorithm.equalsIgnoreCase("FP")) {
            System.out.println("=== Running FP-Growth ===");
            FPGrowth fpGrowth = new FPGrowth();
            fpGrowth.setNumRulesToFind(10); // number of rules to output
            fpGrowth.buildAssociations(newData);

            // Print rules
            AssociationRules rules = fpGrowth.getAssociationRules();
            for (AssociationRule rule : rules.getRules()) {
                System.out.println(rule);
            }
        } else {
            System.out.println("Invalid algorithm choice! Use 'Apriori' or 'FP'.");
        }
    }
}

