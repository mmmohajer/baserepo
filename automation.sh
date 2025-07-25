#!/bin/bash

source ./utils/shellScripting/constants/colours.sh
source ./utils/shellScripting/constants/constants.sh
source ./utils/shellScripting/constants/versioning.sh
source ./utils/shellScripting/funcs/secret_vars.sh
source ./utils/shellScripting/funcs/helpers.sh
source ./utils/shellScripting/funcs/contexts.sh
source ./utils/shellScripting/funcs/client.sh
source ./utils/shellScripting/funcs/api.sh
source ./utils/shellScripting/funcs/cicd.sh
source ./utils/shellScripting/funcs/deploy.sh
source ./utils/shellScripting/funcs/db.sh

cat << EOF
This script runs to help you develop your application much faster.
EOF

showMenuBar

run() {
    local selected
    local isRunning=0
    while [[ isRunning -eq 0 ]]
    do
        read -p "Choose an option (0 to show menubar): " selected
        [ $selected == Q ] && exit 0
        if [[ ${OPTIONS[*]} =~ $selected ]]
        then
            [ $selected == 0 ] && showMenuBar
            [ $selected == 1 ] && createReactComponent
            [ $selected == 2 ] && createReactPage
            [ $selected == 3 ] && addDjangoApp
            [ $selected == 4 ] && deployInLocal
            [ $selected == 5 ] && deployToProdWithSwarm
            [ $selected == 6 ] && deployToProdWithCompose
            [ $selected == 7 ] && makeBackupOfDb
            [ $selected == 8 ] && restoreDb
        else
            run
        fi
    done
    return 0
}

run