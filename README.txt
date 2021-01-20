dialogues should be pre-downloaded with chrome app "VkOpt"

formatted like this:

/./%username%:
%message%

and saved in C:/dialogues by default
(or another path, but if so, the variable "dialoguesPath" in "mainv2.py" needs to be changed)

the folder of the opponent should be named according to the zip file's name
example: "Ivan Ivanov(391726498)"

-----------------------------------------------
in usrdata.py:

change values of "login" and "password"
WARNING: your profile should NOT have double authentification enabled for this to work.

-----------------------------------------------
in mainv2.py:
change value of "vkName" to your lowercase vk name (to do: make the API reach it automatically)