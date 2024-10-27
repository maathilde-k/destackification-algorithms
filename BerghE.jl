function BerghE(F::StackyFan, divlist::Array{Int64,1})

    maximalCones = getMaximalCones(F.fan)
    orderOnMaximalCones = Dict{Vector{Int64},Int64}()
    i=0
    for maximalCone in maximalCones
        orderOnMaximalCones[maximalCone] = i
        i += 1
    end

    cones = getCones(F.fan)
    order = Dict{Vector{Int64},Int64}()
    for cone in cones
        indexMaximalCone = findall(x -> issubset(Set(cone),Set(x)), maximalCones)[1]
        order[cone] = orderOnMaximalCones[maximalCones[indexMaximalCone]]
    end
    
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
        #find the collection of minimal cones with maximal aggregrate 
        conelist = minMaxAggregate(X, distVect)
        # if they equal the set of the maximal cones, we're done
        
        if conelist == nothing
            break
        end
        # otherwise, continue
        # for each cone in the collection

        blowupVectors = Vector{Vector{Int64}}([])
        rayConelist = []
        for cone in conelist
            raysOfCone = [getRays(F.fan)[i] for i in cone]
            push!(rayConelist,raysOfCone)
        end

        for rayCone in rayConelist
            allRays = getRays(X.fan)
            cone = findall(x -> x in rayCone, allRays)
            
            ############################### working on the model
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


            print(modelDist)
            instructions = BerghAmod(modelNewSX, modelDist)[3]

            
            ####################################
        
            # working on the actual stacky fan
            blowupVector = findStackyBarycenter(cone, X)
            blowupVectorPrim = Array{QQFieldElem}(Polymake.common.primitive(blowupVector))
            blowupVectorPrim = map(x -> Int64(numerator(x)), blowupVectorPrim)
            X = stackyBlowup(X,cone,blowupVector)
            distVect[blowupVectorPrim] = 1
            push!(blowupVectors, blowupVectorPrim)

        
            #######################################
        
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

        blowupVectors = unique(blowupVectors)

        for blowupVector in blowupVectors
            AllRays = getRays(X.fan)
            distVectList = Vector{Int64}([0 for i in [1:1:size(AllRays,1);]])
            for i in [1:1:size(AllRays,1);]
                distVectList[i]= distVect[AllRays[i]]
            end

            Dlist = Vector{Int64}([0 for i in [1:1:size(AllRays,1);]])
            for i in [1:1:size(AllRays,1);]
                if Dlist[i] == blowupVector
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
