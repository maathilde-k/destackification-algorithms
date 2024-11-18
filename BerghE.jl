include("toric_helpers.jl")
include("BerghA.jl")
include("BerghD.jl")

"""
    BerghE(F::StackyFan,divlist::Array{Int64,1})

Given a stacky fan F and a vector of booleans divlist representing the distinguished structure, 
returns the result of a sequence of smooth stacky blowups such that the independency index is 0 everywhere at each step.

The algorithm is adapted from Daniel Bergh's [paper on destackification](https://arxiv.org/abs/1409.5713). 

# Examples

"""

function BerghE(F::StackyFan, divlist::Array{Int64,1})
    
    X = deepcopy(F)
    slicedRayMatrix = getRays(X.fan)
    maximalCones = getMaximalCones(X.fan)
    
    # Populate dictionary of divisorial rays (div) from divlist
    distVect=Dict{Vector{QQFieldElem}, Int64}()
    for i in 1:size(slicedRayMatrix,1)
        distVect[slicedRayMatrix[i]]=divlist[i]
    end
    
    j= 0
    while(true)
        ################# E1 : find the collection of minimal cones with maximal aggregrate ##################
        conelist = minMaxAggregate(X, distVect)
        
        # if they equal the set of the maximal cones, we're done
        if conelist == nothing
            break
        end
        
        blowupVectors = Vector{Vector{Int64}}([])
        rayConelist = []
        
        # for each cone in the collection of minimal cones with maximal aggregrate, add its rays to the rayConelist
        for cone in conelist
            raysOfCone = [getRays(F.fan)[i] for i in cone]
            push!(rayConelist,raysOfCone)
        end

        
        #for each set of rays representing a cone in rayConelist
        for rayCone in rayConelist

            ################################# E2 : Blowup the worst locus ##################################
            # create a cone containing all the rays of the rayCone
            allRays = getRays(X.fan)
            cone = findall(x -> x in rayCone, allRays)
            
            # create a model of the stacky fan corresponfing to the cone containing rayCone 
            modelRayMatrix = listToMatrix([getRays(X.fan)[i] for i in cone])#listToMatrix(getRays(X.fan))
            modelCone = Array{Int64}([i for i in [1:1:size(modelRayMatrix,1);]])
            modelConeCopy = [i for i in modelCone]
            modelStacks = [stackyWeights(X)[i] for i in cone]
            modelX = normal_toric_variety(IncidenceMatrix([modelConeCopy]),modelRayMatrix)
            modelSX = addStackStructure(modelX,modelStacks)
            modelBlowupVector = findStackyBarycenter(cone, X)
            modelBlowupVector = map(x -> Int64(numerator(x)), modelBlowupVector)
            push!(blowupVectors,modelBlowupVector)
            modelBlowupVectorPrim = Array{QQFieldElem}(Polymake.common.primitive(modelBlowupVector))
            modelGCD = gcd(modelBlowupVector)
            modelStacks = push!(modelStacks, modelGCD)
            NumOfRays = size(modelCone,1)
            if size(modelCone,1) == 1
                modelNewCone = [i for i in modelCone]
                modelNewRays = modelRayMatrix
            else 
                modelNewCone = [deleteat!(push!(copy(modelCone),NumOfRays+1),i) for i in modelCone]
                modelNewRays = vcat(modelRayMatrix,reshape(modelBlowupVector,1,size(modelBlowupVector,1)))
            end
            modelNewX = normal_toric_variety(IncidenceMatrix([i for i in modelNewCone]),modelNewRays)
            modelNewSX = addStackStructure(modelNewX, modelStacks)
            modelDist = [0 for i in [1:1:size(getRays(modelNewSX.fan),1);]]
            for i in [1:1:size(getRays(modelNewSX.fan),1);]
                if getRays(modelNewSX.fan)[i] == modelBlowupVectorPrim
                    modelDist[i] = 1
                end
            end
            
            ############################## E3: perform toric destackification ##############################
            
            # Perform Algorithm A on the model and record the steps taken for the blowup
            instructions = BerghAmod(modelNewSX, modelDist)[3]

            ######################### E4: perform corresponding stacky blowups  ############################
            
            # working on the actual stacky fan
            # perform the same sequence of blowups on the real stacky fan using instructions

            blowupVector = findStackyBarycenter(cone, X)
            blowupVectorPrim = Array{QQFieldElem}(Polymake.common.primitive(blowupVector))
            blowupVectorPrim = map(x -> Int64(numerator(x)), blowupVectorPrim)
            X = stackyBlowup(X,cone,blowupVector)
            distVect[blowupVectorPrim] = 1
            push!(blowupVectors, blowupVectorPrim)

            listOfRays = getRays(X.fan)
            newDivList = [0 for i in [1:1:size(listOfRays,1);]]
            for i in [1:1:size(listOfRays,1);]
                if listOfRays[i] == blowupVectorPrim
                   newDivList[i] = 1 
                end
            end
            
            X, distVectList = manualBerghA(X, newDivList,instructions)
            newDistVect = Dict{Vector{QQFieldElem}, Int64}()
            
            AllRays = getRays(X.fan)
            for i in [1:1:size(AllRays, 1);]
                newDistVect[AllRays[i]] = distVectList[i]
            end
            distVect = newDistVect
        end
        
        ################### E5: eliminate divisorial index along distinguished divisors #####################

        # To do so, we perform algorithm D repeatedly on the blowupVectors
        
        blowupVectors = unique(blowupVectors)

        for blowupVector in blowupVectors
        
            AllRays = getRays(X.fan)

            # make a list of of distinguished divisors
            distVectList = Vector{Int64}([0 for i in [1:1:size(AllRays,1);]])
            for i in [1:1:size(AllRays,1);]
                distVectList[i] = distVect[AllRays[i]]
            end

            # make a list labelling the blowup vector as the vector whose divisorial index we want to make 0.
            Dlist = Vector{Int64}([0 for i in [1:1:size(AllRays,1);]])
            for i in [1:1:size(AllRays,1);]
                if AllRays[i] == blowupVector
                    Dlist[i] = 1
                end
            end
            
            X, distVect = BerghD(X, distVectList,Dlist)
            
            newDistVect = Dict{Vector{QQFieldElem}, Int64}()
            allrays = getRays(X.fan)
            for i in [1:1:size(allrays, 1);]
                newDistVect[allrays[i]] = distVectList[i]
            end
            distVect = newDistVect
        end
    end
    return X, distVect
end
