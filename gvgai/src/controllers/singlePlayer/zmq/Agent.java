package controllers.singlePlayer.zmq;

import core.competition.Communication;
import core.game.Observation;
import core.game.StateObservation;
import core.player.AbstractPlayer;
import ontology.Types;
import ontology.Types.ACTIONS;
import org.zeromq.ZContext;
import org.zeromq.ZFrame;
import org.zeromq.ZMQ;
import org.zeromq.ZMsg;
import tools.ElapsedCpuTimer;
import tools.Vector2d;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

public class Agent extends AbstractPlayer{

    public static int NUM_ACTIONS;
    public static int ROLLOUT_DEPTH = 10;
    public static double K = Math.sqrt(2);
    public static Types.ACTIONS[] actions;
    
    double lastScore = 0;

    static int numObjectsPerCell = 2;  // Maximum number of objects per cell to pass to the agent.
    
    ZFrame address;
    ZFrame content;
    ZMsg msg;
    
    public static int PORT = Communication.port;
    public static ZMQ.Socket socket = null;
	/**
	 * Initialize all variables for the agent. Setup the zmq server
     * and wait for the client to connect.
	 * @param stateObs Observation of the current state.
     * @param elapsedTimer Timer when the action returned is due.
	 */
	public Agent(StateObservation stateObs, ElapsedCpuTimer elapsedTimer){
		//Get the actions in a static array.
		ArrayList<ACTIONS> act = stateObs.getAvailableActions(false);  // true - include NIL

        actions = new Types.ACTIONS[act.size()];
        for(int i = 0; i < actions.length; ++i)
		{
			actions[i] = act.get(i);
		}
		NUM_ACTIONS = actions.length;

        // Send the number of available actions to the agent.
        System.out.println("actions: " + NUM_ACTIONS);
        String outMsg = ""+NUM_ACTIONS;
        socket.send(outMsg);

        // Get confirmation of receipt from agent.
        // Receive action from agent.
        msg = ZMsg.recvMsg(socket);
        ZFrame content = msg.pop();
        assert(content.toString() == "ACK");
	}


    /**
     * Picks an action. This function is called every game step to request an
     * action from the player.
     * @param stateObs Observation of the current state.
     * @param elapsedTimer Timer when the action returned is due.
     * @return An action for the current state
     */
    public ACTIONS act(StateObservation stateObs, ElapsedCpuTimer elapsedTimer) {
    	
    	// state
        ArrayList<Observation> obs[] = stateObs.getFromAvatarSpritesPositions();
        ArrayList<Observation> grid[][] = stateObs.getObservationGrid();
        
        Vector2d pos = stateObs.getAvatarPosition(); // check if already included in getObservation grid
        int health = stateObs.getAvatarHealthPoints();
        HashMap<Integer, Integer> resources = stateObs.getAvatarResources();
        
        // reward
        double currentScore = stateObs.getGameScore();
        double reward = currentScore - lastScore;
        lastScore = currentScore;

        // penalize each step.
        reward -= 0.01;

        // debug information

//        double blsize = stateObs.getBlockSize();
//        System.out.println(stateObs.getBlockSize());
//        System.out.println(stateObs.getWorldDimension());
//        System.out.println(grid.length);
//        System.out.println(grid[0].length);
//        System.out.println("#########################");
//        for(int j = 0; j < grid[0].length; ++j) {
//            for(int i = 0; i < grid.length; ++i) {
//                int n = grid[i][j].size();
//                if(n > 0)
//                    System.out.print(n);
//                else
//                    System.out.print(' ');
//            }
//            System.out.println();
//        }
//        ArrayList<Observation> obs0 = grid[0][0];
//        System.out.println("#########################");
//
//        System.out.println("Grid0,0");
//        for(int j = 0; j < obs0.size(); ++j) {
//        	Observation pobs = obs0.get(j);
//        	System.out.println(pobs.category);
//        	System.out.println(pobs.itype);
//        	System.out.println(pobs.position);
//        }
//        System.out.println("#########################");
//
//        System.out.println(pos);
//        System.out.println("Gridpos");
//        obs0 = grid[(int) (pos.x / blsize)][(int) (pos.y / blsize)];
//
//        for(int j = 0; j < obs0.size(); ++j) {
//        	Observation pobs = obs0.get(j);
//        	System.out.println(pobs.category);
//        	System.out.println(pobs.itype);
//        	System.out.println(pobs.position);
//        }
//        System.out.println("#########################");
//
//        System.out.println(health);
//        System.out.println(resources);
//        System.out.println(currentScore);
//        System .out.println(lastScore);
//        System.out.println(terminal);

        String stateMsg = createStateMsg(stateObs);

        // Send message to agent.
        String outMsg = "state, reward, terminal = "+stateMsg+", "+reward+", false";
        socket.send(outMsg);


        // Receive action from agent.
        msg = ZMsg.recvMsg(socket);
        ZFrame content = msg.pop();

        int actionID;

        try {
            actionID = Integer.parseInt(content.toString());
        } catch (Exception e) {
            socket.close();
            throw e;
        }

        return actions[actionID];
    }

    /**
     * Function called when the game is over. This method must finish before CompetitionParameters.TEAR_DOWN_TIME,
     *  or the agent will be DISQUALIFIED
     * @param stateObservation the game state at the end of the game
     * @param elapsedCpuTimer timer when this method is meant to finish.
     */
    public void result(StateObservation stateObservation, ElapsedCpuTimer elapsedCpuTimer)
    {
        String stateMsg = createStateMsg(stateObservation);

        // reward
        double currentScore = stateObservation.getGameScore();
        double reward = currentScore - lastScore;

        // Check if the game was won or lost - provide appropriate reward.
//        if(stateObservation.getGameWinner() == Types.WINNER.PLAYER_LOSES) {
//            reward -= 100;
//        }

        // Send message to agent.
        String outMsg = "state, reward, terminal = "+stateMsg+", "+reward+", true";
        lastScore = 0; // Needed?
        socket.send(outMsg);
    }

    /**
     * Function to create the state message given the observation grid.
     */
    public String createStateMsg(StateObservation stateObs) {
        ArrayList<Observation> grid[][] = stateObs.getObservationGrid();

        // Create state part of message.
        String stateMsg = "{";
        String [] partMsg = new String[numObjectsPerCell];

        for (int i=0; i< numObjectsPerCell; ++i) {
            partMsg[i] = "{";
        }

        for(int i = 0; i < grid.length; ++i) {
            for (int k = 0; k < numObjectsPerCell; k++)
                partMsg[k] += "{";

            for(int j = 0; j < grid[i].length; ++j) {
                int n = grid[i][j].size();
                if (n > 0) {
                    HashSet<Integer> itypeSet = new HashSet<>();
                    for (int k = n-1; k>=0; k--) {
                        if(itypeSet.size() >= numObjectsPerCell)
                            break;

                        int tmpID = grid[i][j].get(k).objectID;
                        if(tmpID == -1)
                            tmpID = grid[i][j].get(k).itype;
                        if (!itypeSet.contains(tmpID)) {
                            partMsg[itypeSet.size()] += tmpID + ",";
                            itypeSet.add(tmpID);
                        }
                    }

                    for (int k = itypeSet.size(); k < numObjectsPerCell; k++)
                        partMsg[k] += "0,";

                } else {
                    for (int k = 0; k < numObjectsPerCell; k++)
                        partMsg[k] += "0,";
                }

            }
            for (int k = 0; k < numObjectsPerCell; k++)
                partMsg[k] += "},";
        }

        // Merge the two part messages into one stateMsg.
        for (int k = 0; k < numObjectsPerCell; k++)
            stateMsg += partMsg[k] + "},";

        stateMsg += "}";

        return stateMsg;
    }

}
