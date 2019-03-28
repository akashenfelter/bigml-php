<?php

# Copyright 2012-2018 BigML
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

/* A local Fusion object.

   This module defines a Fusion to make predictions locally using its
associated models.  This module can not only save you a few credits,
but also enormously reduce the latency for each prediction and let you
use your models offline.

Example usage (assuming that you have previously set up the
BIGML_USERNAME and BIGML_API_KEY environment variables and that you
own the model/id below): 

include bigml.php;
include fusion.php;

use BigML\BigML;
use BigML\BigMLRequest;
use BigML\Fusion;

// api connection
$api = new BigML();
// creating fusion
$fusion = $api->create_fusion( array( "model/5143a51a37203f2cf7000972",
                                      "model/5143a51a37203f2cf7000985"));
// Fusion object to predict
$local_fusion = new Fusion($fusion, $api);
$local_fusion->predict(array("petal length" => 3, "petal width" => 1));

 */

namespace BigML;

if (!class_exists('BigML\BigML')) {
   include('bigml.php');
}

if (!class_exists('BigML\BaseModel')) {
  include('basemodel.php');
}

if (!class_exists('BigML\ModelFields')) {
  include('modelfields.php');
}

if (!class_exists('BigML\Model')) {
   include('model.php');
}


if (!class_exists('BigML\Ensemble')) {
   include('ensemble.php');
}

if (!class_exists('BigML\LogisticRegression')) {
   include('logistic.php');
}

if (!class_exists('BigML\Deepnet')) {
    include('deepnet.php');
}

use BigML\ModelFields;

class Fusion extends ModelFields{
    /*
      A lightweight wrapper around a Fusion model.  

      Uses a BigML remote model to build a local version that can be
      used to generate predictions locally.

      AKA -- python bindings include caching
    */

    public $model_ids = [];
    public $weights = [];
    public $objective_id = null;
    public $distribution = null;
    public $model_list = [];
    public $regression = false;
    public $fields = null;
    public $class_names = null;
    public $importance = [];

    public function __construct($fusion, $api=null, $storage="storage") {

//          The Fusion constructor can be given as first argument:
//             - a fusion structure
//             - a fusion id

        if ($api == null) {
            $api = new BigML(null, null, null, $storage);
        }
        
        if (is_string($fusion)) {
            if (!($api->_checkFusionId($fusion)) ) {
                error_log("Wrong fusion id");
                return null;
            } else {
                $fusion = $api->retrieve_resource($fusion, 
                                                   BigML::ONLY_MODEL);
            }
        }

        if (property_exists($fusion, "object") && 
            property_exists($fusion->object, "status") && 
            $fusion->object->status->code != BigMLRequest::FINISHED ) {
            throw new \Exception("The fusion isn't finished yet");
        }

        if (property_exists($fusion, "object") && 
            $fusion->object instanceof \STDClass) {
            $fusion = $fusion->object;

            $models_info = $fusion->models;
            $model_ids = [];
            $weights = [];
            if (is_array($models_info[0])) {
                foreach ($models_info as $model) {
                    $model_ids[] = $model["id"];
                    $weights[] = $model["weight"];
                }
            } else {
                $model_ids = $models_info;
                $weights = null;
            }
            $this->model_ids = $model_ids;
            $this->weights = $weights;

            if (array_key_exists("importance", $fusion)) {
                $this->importance = $fusion->importance;
            } else {
                $this->importance = [];
            }

            if (array_key_exists("missing_numerics", $fusion)) {
                $this->missing_numerics = $fusion->missing_numerics;
            } else {
                $this->missing_numerics = true;
            }
        }

        if (property_exists($fusion, "fusion") && 
            $fusion->fusion instanceof \STDClass) {

            if ($fusion->status->code == BigMLRequest::FINISHED) {

                $objective_id = extract_objective($fusion->objective_fields);
                $fusion = $fusion->fusion;

                $this->fields = $fusion->fields;
                parent::__construct($this->fields, 
                                    $objective_id, 
                                    null, null, true, true);

                $this->regression = ($this->fields->$objective_id->optype ==
                                     NUMERIC);
                if (!$this->regression) {
                    foreach ($this->fields->$objective_id->summary->categories as $category) {
                        $this->class_names[] = $category[0];
                    }
                    sort($this->class_names);
                }

                foreach ($this->model_ids as $model) {
                    if (preg_match('/model/', $model) ) {
                      $new_model = new Model($model); 
                    } elseif (preg_match('/ensemble/', $model) ) {
                      $new_model = new Ensemble($model);  
                    } elseif (preg_match('/logisticregression/', $model) ) {
                      $new_model = new LogisticRegression($model);  
                    } elseif (preg_match('/deepnet/', $model) ) {
                      $new_model = new Deepnet($model);  
                    }
                    $model_list[] = $new_model;
                }

                $this->model_list = $model_list;

                if (isset($this->fields)) {
                    $summary = $this->fields->$objective_id->summary;
                    if (array_key_exists("bins", $summary)) {
                        $distribution = $summary->bins;
                    } elseif (array_key_exists("counts", $summary)) {
                        $distribution = $summary->counts;
                    } elseif (array_key_exists("categories", $summary)) {
                        $distribution = $summary->categories;
                    } else {
                        $distribution = [];
                    }
                    $this->distribution = $distribution;
                }

            } else {
                throw new \Exception("The fusion isn't finished yet");
            }
        } else {
            throw new \Exception("Cannot create the Fusion instance. Could not find the 'fusion' key in the resource.\n\n ");
        }
    }

   public function predict_probability($input_data, $by_name=true, $compact=false) {
         // Predicts a probability for each possible output class,
         // based on input values.  The input fields must be a dictionary
         // keyed by field name or field ID.
 
         // :param input_data: Input data to be predicted
         // :param by_name: Boolean that is set to True if field_names (as
         //                 alternative to field ids) are used in the
         //                 input_data dict
         // :param compact: If False, prediction is returned as a list of maps, one
         //                 per class, with the keys "prediction" and "probability"
         //                 mapped to the name of the class and it's probability,
         //                 respectively.  If True, returns a list of probabilities
         //                 ordered by the sorted order of the class names.
   }


}

?>
