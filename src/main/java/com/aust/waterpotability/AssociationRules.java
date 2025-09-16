package com.aust.waterpotability;

import weka.associations.Apriori;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class AssociationRules {
    public static void main(String[] args) throws Exception {
        Instances data = DataSource.read("data/processed_water_potability.arff");

        // Apriori expects nominal attributes
        Apriori model = new Apriori();
        model.setClassIndex(data.numAttributes() - 1); // Potability
        model.buildAssociations(data);

        System.out.println("===== Association Rules (Apriori) =====");
        System.out.println(model);
    }
}
