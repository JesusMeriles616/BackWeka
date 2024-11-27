package com.example.demo.service;

import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.clusterers.SimpleKMeans;

import java.io.InputStream;
import java.io.IOException;

@Service
public class WekaService {

    public String analizarArchivo(MultipartFile file) throws IOException {
        try {
            // Cargar archivo CSV o ARFF
            InputStream inputStream = file.getInputStream();
            DataSource source = new DataSource(inputStream);
            Instances data = source.getDataSet();

            // Establecer el índice de clase si no se ha definido
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }

            // Realizar análisis utilizando K-Means (Ejemplo)
            SimpleKMeans kMeans = new SimpleKMeans();
            kMeans.setNumClusters(3); // Número de clusters
            kMeans.buildClusterer(data);

            // Obtener el número de clusters y otras estadísticas
            return "Número de clusters: " + kMeans.getNumClusters();
        } catch (Exception e) {
            e.printStackTrace();
            return "Error en el análisis del archivo.";
        }
    }



}
