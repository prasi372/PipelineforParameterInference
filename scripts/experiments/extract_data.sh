#Usage: ./extract_data.sh <folders>
for path in `cat $1`; do
    folder=`dirname $path`
    exp=`basename $path`
    echo "Extracting for $path"
    #python3 extract_model_expvalue.py $folder/expvalue_$exp.json $path/inference_*_smoldyn.db $path/inference_*_CBM.db $path/inference_*_WMM.db
    #python3 extract_model_logexpvalue.py $folder/logexpvalue_$exp.json $path/inference_*_smoldyn.db $path/inference_*_CBM.db $path/inference_*_WMM.db
    python3 extract_model_logerror.py $folder/logerror_$exp.json $path/inference_*_smoldyn.db $path/inference_*_CBM.db $path/inference_*_WMM.db
    #python3 extract_model_likelihood.py $folder/lklh_$exp.json $path/inference_*_smoldyn.db $path/inference_*_CBM.db $path/inference_*_WMM.db
    #python3 extract_model_prob.py $folder/mprob_$exp.json $path/inference_*_All.db
    #python3 extract_model_utility.py $folder/utility_$exp.json $path/inference_*_log.json
    echo ""
done
