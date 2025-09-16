package com.aust.waterpotability;

import weka.core.Instances;

public class Main {
    public static void main(String[] args) throws Exception {
        System.out.println("=== Water Potability Project ===");

        Instances data = Utils.loadDataset("data/processed_water_potability.arff");
        Utils.printDatasetInfo(data);

        System.out.println("\n--- Running Supervised Models ---");
        SupervisedModel.main(args);

        System.out.println("\n--- Running Unsupervised Models ---");
        UnsupervisedModel.main(args);

        System.out.println("\n--- Running Association Rules ---");
        AssociationRulesRunner.main(args);

        System.out.println("=== Done ===");
    }
}
