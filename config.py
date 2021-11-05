bert_dir = '../BertModel/bert-base-cased/'

event_schema = {'life.die.deathcausedbyviolentevents': {'killer', 'place', 'instrument', 'victim'}, 'movement.transportartifact.hide': {'vehicle', 'artifact', 'transporter', 'origin', 'hidingplace'}, 'conflict.attack.selfdirectedbattle': {'target', 'place', 'attacker', 'instrument'}, 'life.injure.illnessdegradationphysical': {'victim'}, 'contact.commitmentpromiseexpressintent.n/a': {'recipient', 'place', 'communicator'}, 'justice.arrestjaildetain.arrestjaildetain': {'jailer', 'place', 'detainee', 'crime'}, 'contact.discussion.meet': {'participant', 'place'}, 'life.injure.injurycausedbyviolentevents': {'place', 'instrument', 'victim', 'injurer'}, 'personnel.endposition.n/a': {'place', 'placeofemployment', 'employee'}, 'transaction.transferownership.n/a': {'recipient', 'giver', 'place', 'beneficiary', 'artifact'}, 'justice.investigate.investigatecrime': {'place', 'crime', 'investigator', 'defendant'}, 'contact.collaborate.n/a': {'participant', 'place'}, 'government.agreements.violateagreement': {'otherparticipant', 'violator', 'place'}, 'movement.transportperson.prevententry': {'preventer', 'destination', 'transporter', 'origin', 'passenger'}, 'contact.commandorder.broadcast': {'recipient', 'place', 'communicator'}, 'transaction.transfermoney.n/a': {'recipient', 'giver', 'place', 'beneficiary', 'money'}, 'justice.initiatejudicialprocess.n/a': {'place', 'crime', 'defendant', 'judgecourt', 'prosecutor'}, 'contact.prevarication.broadcast': {'recipient', 'place', 'communicator'}, 'conflict.attack.stealrobhijack': {'place', 'instrument', 'target', 'artifact', 'attacker'}, 'life.injure.illnessdegradationhungerthirst': {'place', 'victim'}, 'contact.negotiate.meet': {'participant', 'place'}, 'contact.threatencoerce.n/a': {'recipient', 'place', 'communicator'}, 'contact.commitmentpromiseexpressintent.broadcast': {'recipient', 'place', 'communicator'}, 'personnel.elect.n/a': {'voter', 'place', 'candidate'}, 'transaction.transfermoney.purchase': {'recipient', 'giver', 'place', 'beneficiary', 'money'}, 'contact.mediastatement.broadcast': {'recipient', 'place', 'communicator'}, 'contact.requestadvise.correspondence': {'recipient', 'place', 'communicator'}, 'movement.transportartifact.disperseseparate': {'vehicle', 'destination', 'artifact', 'transporter', 'origin'}, 'government.legislate.legislate': {'law', 'place', 'governmentbody'}, 'movement.transportperson.preventexit': {'preventer', 'destination', 'transporter', 'origin', 'passenger'}, 'contact.negotiate.n/a': {'participant', 'place'}, 'government.agreements.n/a': {'participant', 'place'}, 'life.injure.n/a': {'place', 'victim', 'injurer'}, 'justice.judicialconsequences.extradite': {'crime', 'extraditer', 'defendant', 'destination', 'origin'}, 'personnel.endposition.firinglayoff': {'place', 'placeofemployment', 'employee'}, 'justice.investigate.n/a': {'place', 'investigator', 'defendant'}, 'movement.transportartifact.sendsupplyexport': {'vehicle', 'destination', 'artifact', 'transporter', 'origin'}, 'government.agreements.acceptagreementcontractceasefire': {'participant', 'place'}, 'disaster.fireexplosion.fireexplosion': {'place', 'instrument', 'fireexplosionobject'}, 'contact.collaborate.correspondence': {'participant', 'place'}, 'transaction.transaction.transfercontrol': {'recipient', 'giver', 'place', 'beneficiary', 'territoryorfacility'}, 'transaction.transfermoney.giftgrantprovideaid': {'recipient', 'giver', 'place', 'beneficiary', 'money'}, 'movement.transportartifact.nonviolentthrowlaunch': {'vehicle', 'destination', 'artifact', 'transporter', 'origin'}, 'contact.commitmentpromiseexpressintent.correspondence': {'recipient', 'place', 'communicator'}, 'conflict.attack.airstrikemissilestrike': {'target', 'place', 'attacker', 'instrument'}, 'government.formation.n/a': {'place', 'gpe', 'founder'}, 'movement.transportperson.hide': {'vehicle', 'transporter', 'origin', 'hidingplace', 'passenger'}, 'justice.judicialconsequences.execute': {'place', 'executioner', 'crime', 'defendant'}, 'transaction.transaction.embargosanction': {'recipient', 'giver', 'place', 'preventer', 'artifactmoney'}, 'conflict.attack.stabbing': {'target', 'place', 'attacker', 'instrument'}, 'conflict.yield.retreat': {'retreater', 'destination', 'origin'}, 'transaction.transfermoney.embargosanction': {'recipient', 'giver', 'place', 'preventer', 'money'}, 'manufacture.artifact.build': {'place', 'manufacturer', 'instrument', 'artifact'}, 'inspection.sensoryobserve.n/a': {'observer', 'place', 'observedentity'}, 'justice.initiatejudicialprocess.trialhearing': {'place', 'crime', 'defendant', 'judgecourt', 'prosecutor'}, 'movement.transportartifact.smuggleextract': {'vehicle', 'destination', 'artifact', 'transporter', 'origin'}, 'contact.requestadvise.broadcast': {'recipient', 'place', 'communicator'}, 'contact.commitmentpromiseexpressintent.meet': {'recipient', 'place', 'communicator'}, 'government.spy.spy': {'spy', 'beneficiary', 'place', 'observedentity'}, 'government.vote.violationspreventvote': {'place', 'preventer', 'candidate', 'ballot', 'voter'}, 'contact.discussion.n/a': {'participant', 'place'}, 'contact.commandorder.correspondence': {'recipient', 'place', 'communicator'}, 'justice.judicialconsequences.n/a': {'defendant', 'place', 'crime', 'judgecourt'}, 'conflict.attack.firearmattack': {'target', 'place', 'attacker', 'instrument'}, 'contact.prevarication.correspondence': {'recipient', 'place', 'communicator'}, 'movement.transportartifact.bringcarryunload': {'vehicle', 'destination', 'artifact', 'transporter', 'origin'}, 'conflict.attack.strangling': {'target', 'place', 'attacker', 'instrument'}, 'contact.requestadvise.n/a': {'recipient', 'place', 'communicator'}, 'artifactexistence.damagedestroy.destroy': {'destroyer', 'place', 'instrument', 'artifact'}, 'life.die.n/a': {'place', 'victim'}, 'contact.threatencoerce.meet': {'recipient', 'place', 'communicator'}, 'personnel.startposition.hiring': {'place', 'placeofemployment', 'employee'}, 'conflict.attack.n/a': {'target', 'place', 'attacker', 'instrument'}, 'personnel.endposition.quitretire': {'place', 'placeofemployment', 'employee'}, 'justice.initiatejudicialprocess.chargeindict': {'place', 'crime', 'defendant', 'judgecourt', 'prosecutor'}, 'contact.requestadvise.meet': {'recipient', 'place', 'communicator'}, 'government.formation.startgpe': {'place', 'gpe', 'founder'}, 'transaction.transfermoney.payforservice': {'recipient', 'giver', 'place', 'beneficiary', 'money'}, 'personnel.elect.winelection': {'voter', 'place', 'candidate'}, 'movement.transportperson.grantentryasylum': {'granter', 'destination', 'transporter', 'origin', 'passenger'}, 'movement.transportartifact.n/a': {'vehicle', 'destination', 'artifact', 'transporter', 'origin'}, 'contact.publicstatementinperson.broadcast': {'recipient', 'place', 'communicator'}, 'contact.discussion.correspondence': {'participant', 'place'}, 'movement.transportperson.disperseseparate': {'vehicle', 'destination', 'transporter', 'origin', 'passenger'}, 'transaction.transferownership.purchase': {'recipient', 'giver', 'place', 'beneficiary', 'artifact'}, 'movement.transportperson.n/a': {'vehicle', 'destination', 'transporter', 'origin', 'passenger'}, 'conflict.demonstrate.n/a': {'demonstrator', 'place'}, 'conflict.demonstrate.marchprotestpoliticalgathering': {'demonstrator', 'place'}, 'movement.transportperson.smuggleextract': {'vehicle', 'destination', 'transporter', 'origin', 'passenger'}, 'inspection.sensoryobserve.physicalinvestigateinspect': {'inspector', 'place', 'inspectedentity'}, 'contact.publicstatementinperson.n/a': {'recipient', 'place', 'communicator'}, 'justice.judicialconsequences.convict': {'defendant', 'place', 'crime', 'judgecourt'}, 'contact.funeralvigil.meet': {'deceased', 'participant', 'place'}, 'manufacture.artifact.createmanufacture': {'place', 'manufacturer', 'instrument', 'artifact'}, 'conflict.yield.n/a': {'recipient', 'place', 'yielder'}, 'government.formation.mergegpe': {'participant', 'place'}, 'transaction.transfermoney.borrowlend': {'recipient', 'giver', 'place', 'beneficiary', 'money'}, 'transaction.transaction.n/a': {'beneficiary', 'participant', 'place'}, 'transaction.transferownership.embargosanction': {'recipient', 'place', 'giver', 'preventer', 'artifact'}, 'transaction.transaction.giftgrantprovideaid': {'beneficiary', 'recipient', 'place', 'giver'}, 'artifactexistence.damagedestroy.damage': {'place', 'instrument', 'artifact', 'damager'}, 'contact.prevarication.n/a': {'recipient', 'place', 'communicator'}, 'government.vote.n/a': {'place', 'candidate', 'ballot', 'result', 'voter'}, 'conflict.attack.invade': {'target', 'place', 'attacker', 'instrument'}, 'contact.threatencoerce.correspondence': {'recipient', 'place', 'communicator'}, 'personnel.startposition.n/a': {'place', 'placeofemployment', 'employee'}, 'contact.funeralvigil.n/a': {'deceased', 'participant', 'place'}, 'contact.threatencoerce.broadcast': {'recipient', 'place', 'communicator'}, 'conflict.attack.biologicalchemicalpoisonattack': {'target', 'place', 'attacker', 'instrument'}, 'conflict.attack.bombing': {'target', 'place', 'attacker', 'instrument'}, 'life.die.nonviolentdeath': {'place', 'victim'}, 'contact.collaborate.meet': {'participant', 'place'}, 'contact.negotiate.correspondence': {'participant', 'place'}, 'government.agreements.rejectnullifyagreementcontractceasefire': {'otherparticipant', 'rejecternullifier', 'place'}, 'disaster.accidentcrash.accidentcrash': {'vehicle', 'driverpassenger', 'place', 'crashobject'}, 'transaction.transferownership.borrowlend': {'recipient', 'place', 'giver', 'beneficiary', 'artifact'}, 'movement.transportartifact.preventexit': {'preventer', 'destination', 'artifact', 'transporter', 'origin'}, 'movement.transportperson.evacuationrescue': {'vehicle', 'destination', 'transporter', 'origin', 'passenger'}, 'movement.transportartifact.receiveimport': {'vehicle', 'destination', 'artifact', 'transporter', 'origin'}, 'artifactexistence.damagedestroy.n/a': {'instrument', 'place', 'damagerdestroyer', 'artifact'}, 'movement.transportartifact.grantentry': {'transporter', 'origin', 'destination', 'artifact'}, 'government.vote.castvote': {'place', 'candidate', 'ballot', 'result', 'voter'}, 'contact.commandorder.meet': {'recipient', 'place', 'communicator'}, 'conflict.yield.surrender': {'surrenderer', 'recipient', 'place'}, 'contact.commandorder.n/a': {'recipient', 'place', 'communicator'}, 'contact.prevarication.meet': {'recipient', 'place', 'communicator'}, 'transaction.transferownership.giftgrantprovideaid': {'recipient', 'giver', 'place', 'beneficiary', 'artifact'}, 'manufacture.artifact.n/a': {'place', 'manufacturer', 'instrument', 'artifact'}, 'inspection.sensoryobserve.inspectpeopleorganization': {'inspector', 'place', 'inspectedentity'}, 'movement.transportperson.bringcarryunload': {'vehicle', 'destination', 'transporter', 'origin', 'passenger'}, 'movement.transportartifact.fall': {'origin', 'destination', 'artifact'}, 'inspection.sensoryobserve.monitorelection': {'monitoredentity', 'place', 'monitor'}, 'movement.transportperson.fall': {'origin', 'passenger', 'destination'}, 'manufacture.artifact.createintellectualproperty': {'place', 'manufacturer', 'instrument', 'artifact'}, 'contact.mediastatement.n/a': {'recipient', 'place', 'communicator'}, 'conflict.attack.hanging': {'target', 'place', 'attacker', 'instrument'}, 'movement.transportartifact.prevententry': {'preventer', 'destination', 'artifact', 'transporter', 'origin'}, 'conflict.attack.setfire': {'target', 'place', 'attacker', 'instrument'}, 'movement.transportperson.selfmotion': {'transporter', 'origin', 'destination'}}

